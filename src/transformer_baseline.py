"""Comparable BERT, RoBERTa, and DeBERTa baseline experiments on frozen splits."""
import argparse, json, os, random, time
from datetime import UTC, datetime
from pathlib import Path
import numpy as np, pandas as pd, torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from torch.utils.data import Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                          EarlyStoppingCallback, Trainer, TrainingArguments, set_seed)

MODELS = {"bert":"bert-base-uncased", "roberta":"roberta-base", "deberta":"microsoft/deberta-v3-base"}
LABELS = [0,1,2]; LABEL_NAMES = ["Legitimate","Human-written Fraud","AI-generated Fraud"]

class TextDataset(Dataset):
    def __init__(self, encodings, labels): self.encodings, self.labels = encodings, list(map(int, labels))
    def __len__(self): return len(self.labels)
    def __getitem__(self, index): return {key: torch.tensor(value[index]) for key,value in self.encodings.items()} | {"labels":torch.tensor(self.labels[index])}

def configure(seed):
    os.environ["PYTHONHASHSEED"] = str(seed); random.seed(seed); np.random.seed(seed); set_seed(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

def load_splits(root):
    frames = {name:pd.read_csv(root/f"data/splits/{name}.csv", dtype=str, keep_default_na=False) for name in ("train","validation","test")}
    for name, frame in frames.items():
        if set(frame.label.astype(int)) - set(LABELS): raise ValueError(f"Unexpected labels in {name}")
        if not frame.model_input_text.str.strip().all(): raise ValueError(f"Empty model input in {name}")
    return frames

def metrics(prediction):
    logits, labels = prediction; predicted = np.argmax(logits, axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(labels,predicted,labels=LABELS,zero_division=0)
    macro = precision_recall_fscore_support(labels,predicted,labels=LABELS,average="macro",zero_division=0)
    weighted = precision_recall_fscore_support(labels,predicted,labels=LABELS,average="weighted",zero_division=0)
    return {"accuracy":accuracy_score(labels,predicted), "macro_precision":macro[0],"macro_recall":macro[1],"macro_f1":macro[2],"weighted_precision":weighted[0],"weighted_recall":weighted[1],"weighted_f1":weighted[2], **{f"class_{label}_{key}":float(value) for label,p,r,f,s in zip(LABELS,precision,recall,f1,support) for key,value in {"precision":p,"recall":r,"f1":f,"support":s}.items()}}

def run(model_key, root, epochs=3, batch_size=8, max_length=256, learning_rate=2e-5, seed=42):
    if model_key not in MODELS: raise ValueError(f"Choose one of {sorted(MODELS)}")
    root=Path(root).resolve(); configure(seed); frames=load_splits(root); checkpoint=MODELS[model_key]
    output=root/f"models/{model_key}/baseline_v1"; output.mkdir(parents=True,exist_ok=True)
    tokenizer=AutoTokenizer.from_pretrained(checkpoint)
    datasets={name:TextDataset(tokenizer(frame.model_input_text.tolist(),truncation=True,max_length=max_length),frame.label) for name,frame in frames.items()}
    model=AutoModelForSequenceClassification.from_pretrained(checkpoint,num_labels=3,id2label=dict(enumerate(LABEL_NAMES)),label2id={name:index for index,name in enumerate(LABEL_NAMES)})
    args=TrainingArguments(output_dir=str(output),learning_rate=learning_rate,per_device_train_batch_size=batch_size,per_device_eval_batch_size=batch_size,num_train_epochs=epochs,weight_decay=0.01,eval_strategy="epoch",save_strategy="epoch",logging_strategy="epoch",load_best_model_at_end=True,metric_for_best_model="macro_f1",greater_is_better=True,save_total_limit=2,report_to=[],seed=seed,data_seed=seed,fp16=torch.cuda.is_available())
    trainer=Trainer(model=model,args=args,train_dataset=datasets["train"],eval_dataset=datasets["validation"],processing_class=tokenizer,data_collator=DataCollatorWithPadding(tokenizer),compute_metrics=metrics,callbacks=[EarlyStoppingCallback(early_stopping_patience=2)])
    started=time.time(); trainer.train(); validation=trainer.evaluate(datasets["validation"],metric_key_prefix="validation"); test_output=trainer.predict(datasets["test"]); test=metrics((test_output.predictions,test_output.label_ids)); trainer.save_model(str(output/"final_checkpoint")); tokenizer.save_pretrained(str(output/"final_checkpoint"))
    cm=confusion_matrix(test_output.label_ids,np.argmax(test_output.predictions,axis=1),labels=LABELS).tolist()
    config={"model":model_key,"checkpoint":checkpoint,"tokenizer":tokenizer.name_or_path,"seed":seed,"learning_rate":learning_rate,"batch_size":batch_size,"epochs":epochs,"max_length":max_length,"optimizer":"AdamW","scheduler":"linear","early_stopping_patience":2,"class_weighting":"none (baseline)","device":"cuda" if torch.cuda.is_available() else "cpu","torch":torch.__version__,"transformers":__import__('transformers').__version__,"duration_seconds":time.time()-started,"class2_limitation":"Class 2 has 4 total examples: 2 train, 1 validation, 1 test; its metrics are highly unstable."}
    results=root/"results"; (results/"metrics").mkdir(exist_ok=True); (results/"confusion_matrices").mkdir(exist_ok=True)
    payload={"configuration":config,"validation":validation,"test":test,"confusion_matrix":cm,"labels":dict(zip(LABELS,LABEL_NAMES))}
    (results/f"metrics/{model_key}_metrics.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6,5)); plt.imshow(cm,cmap="Blues"); plt.xticks(range(3),LABEL_NAMES,rotation=25,ha="right"); plt.yticks(range(3),LABEL_NAMES); plt.xlabel("Predicted"); plt.ylabel("True")
    for i in range(3):
        for j in range(3): plt.text(j,i,cm[i][j],ha="center",va="center")
    plt.tight_layout(); plt.savefig(results/f"confusion_matrices/{model_key}_confusion_matrix.png",dpi=160); plt.close()
    return payload

if __name__ == "__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("model",choices=MODELS); parser.add_argument("--root",default=Path(__file__).resolve().parents[1]); parser.add_argument("--epochs",type=int,default=3); parser.add_argument("--batch-size",type=int,default=8); parser.add_argument("--max-length",type=int,default=256); args=parser.parse_args(); print(json.dumps(run(args.model,args.root,args.epochs,args.batch_size,args.max_length),indent=2))
