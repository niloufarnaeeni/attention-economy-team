import torch
from torch import nn
from torch.nn import MSELoss, BCEWithLogitsLoss
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import tqdm
#import ranking_loss
from rag_retrieval.train.reranker import ranking_loss


class CrossEncoder(nn.Module):
    def __init__(
        self,
        hf_model=None,
        tokenizer=None,
        loss_type="point_ce",
        query_format="{}",
        document_format="{}",
    ):
        super().__init__()

        self.model = hf_model
        self.tokenizer = tokenizer
        self.loss_type = loss_type
        self.query_format = query_format
        self.document_format = document_format
        self.train_group_size = None  # to be set in train_reranker.py

    def forward(self, batch, labels=None):
        output = self.model(**batch)
    
        # Robust: avoid squeeze() turning [1,1] into scalar
        logits = output.logits.view(-1)
    
        # If no labels, just return model outputs (inference)
        if labels is None:
            return output
    
        # Support dict targets from GroupedRankerDatasetWithYaps
        yaps = None
        if isinstance(labels, dict):
            yaps = labels.get("yaps", None)
            labels = labels.get("labels", None)
    
        if labels is None:
            raise ValueError("labels is None after unpacking. Check your dataset/collate_fn.")
    
        # ---- ONE loss switch (prevents overwriting) ----
        if self.loss_type == "pairwise_ranknet_yap_neg":
            if yaps is None:
                raise ValueError("pairwise_ranknet_yap_neg requires yaps, but got None.")
    
            lambda_yap = float(getattr(self, "lambda_", 0.3))
    
            loss = ranking_loss.pairwise_ranknet_yap_neg(
                    logits=logits,
                    labels=labels,
                    yaps=yaps,
                    train_group_size=self.train_group_size,
                    lambda_yap=lambda_yap,
                )
        elif self.loss_type == "pairwise_ranknet_yap_both":
            if yaps is None:
                raise ValueError("pairwise_ranknet_yap_both requires yaps, but got None.")
        
            lambda_yap = float(getattr(self, "lambda_", 0.3))
        
            loss = ranking_loss.pairwise_ranknet_yap_both(
                logits=logits,
                labels=labels,
                yaps=yaps,
                train_group_size=self.train_group_size,
                lambda_yap=lambda_yap,
            )

        elif self.loss_type == "pairwise_ranknet_yap_pos":
            if yaps is None:
                raise ValueError("pairwise_ranknet_yap_pos requires yaps")
        
            lambda_yap = float(getattr(self, "lambda_", 0.3))
        
            loss = ranking_loss.pairwise_ranknet_yap_pos(
                logits=logits,
                labels=labels,
                yaps=yaps,
                train_group_size=self.train_group_size,
                lambda_yap=lambda_yap,
            )

        elif self.loss_type == "pairwise_ranknet_centered":
            lambda_centered = float(getattr(self, "lambda_", 0.2))
    
            loss = ranking_loss.ranknet_with_centered_reg(
                logits=logits,
                labels=labels,
                group_size=self.train_group_size,
                lambda_centered=lambda_centered,
            )
    
        elif self.loss_type == "pointwise_mse":
            loss = ranking_loss.pointwise_mse(logits, labels)
    
        elif self.loss_type == "pointwise_bce":
            loss = ranking_loss.pointwise_bce(logits, labels)
    
        elif self.loss_type == "pairwise_ranknet":
            loss = ranking_loss.pairwise_ranknet(logits, labels, self.train_group_size)
    
        elif self.loss_type == "listwise_ce":
            loss = ranking_loss.listwise_ce(logits, labels, self.train_group_size)
        elif self.loss_type == "pairwise_soft_ranknet":
            #tau = float(getattr(self, "tau", 1.0))
        
            loss = ranking_loss.pairwise_soft_ranknet(
                logits=logits,
                labels=labels,
                group_size=self.train_group_size,
                tau=1.0,
            )

    
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")
    
        output["loss"] = loss
        return output


    @torch.no_grad()
    def compute_score(
        self, sentences_pairs, batch_size=256, max_length=512, normalize=False
    ):
        """
        sentences_pairs=[[query,title],[query1,title1],...]
        """

        all_logits = []
        for start_index in tqdm.tqdm(range(0, len(sentences_pairs), batch_size)):
            sentences_batch = sentences_pairs[start_index : start_index + batch_size]
            batch_data = self.preprocess(sentences_batch, max_length).to(self.model.device
            )
            output = self.forward(batch_data)
            logits = output.logits.detach().cpu()
            all_logits.extend(logits)

        if normalize:
            all_logits = torch.sigmoid(torch.tensor(all_logits)).detach().cpu().tolist()

        return all_logits

    def preprocess(self, sentences_pairs, max_len):
        new_sentences_pairs = []
        for query, document in sentences_pairs:
            new_query = self.query_format.format(query.strip())
            new_document = self.document_format.format(document.strip())
            new_sentences_pairs.append([new_query, new_document])
        assert len(new_sentences_pairs) == len(sentences_pairs)

        tokens = self.tokenizer.batch_encode_plus(
            new_sentences_pairs,
            add_special_tokens=True,
            padding="longest",
            max_length=max_len,
            truncation='only_second',
            return_tensors="pt",
        )
        return tokens

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path,
        loss_type="point_ce",
        num_labels=1,
        query_format="{}",
        document_format="{}",
    ):
        hf_model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path, num_labels=num_labels, trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        
        reranker = cls(
            hf_model,
            tokenizer,
            loss_type,
            query_format,
            document_format,
        )
        return reranker

    def save_pretrained(self, save_dir, safe_serialization=False):
        # 模型的参数无论原本是分布在多张卡还是单张卡上，保存后的权重都在 CPU 上，避免了跨设备加载的潜在问题。
        def _trans_state_dict(state_dict):
            state_dict = type(state_dict)(
                {k: v.clone().cpu() for k, v in state_dict.items()}
            )
            return state_dict

        self.model.save_pretrained(
            save_dir,
            state_dict=_trans_state_dict(self.model.state_dict()),
            safe_serialization=safe_serialization,
        )


def test_CrossEncoder():
    ckpt_path = "./bge-reranker-m3-base"
    reranker = CrossEncoder.from_pretrained(
        model_name_or_path=ckpt_path,
        num_labels=1,  # binary classification
    )
    reranker.model.to("cuda:0")
    reranker.eval()
    

    input_lst = [
        ["我喜欢中国", "我喜欢中国"],
        ["我喜欢美国", "我一点都不喜欢美国"],
        [
            "泰山要多长时间爬上去",
            "爬上泰山需要1-8个小时，具体的时间需要看个人的身体素质。专业登山运动员可能只需要1个多小时就可以登顶，有些身体素质比较低的，爬的慢的就需要5个多小时了。",
        ],
    ]

    res = reranker.compute_score(input_lst)

    print(torch.sigmoid(res[0]))
    print(torch.sigmoid(res[1]))
    print(torch.sigmoid(res[2]))


if __name__ == "__main__":
    test_CrossEncoder()
