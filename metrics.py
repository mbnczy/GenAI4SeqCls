import numpy as np
from sklearn.metrics import accuracy_score, classification_report, matthews_corrcoef, cohen_kappa_score

def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)

def hallucination_rate(preds, valid_labels):
    hallucinations = [pred for pred in preds if int(pred) not in valid_labels]
    return len(hallucinations) / len(preds)

def compute_cls_metrics(eval_preds, true_labels, valid_labels, tokenizer):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)

    last_preds = []
    
    for i in range(preds.shape[0]):
        last_pred_idx = np.where(np.logical_and(preds[i] != tokenizer.pad_token_id,preds[i] != 271,preds[i] != 512))[0][-1]
        
        last_preds.append(preds[i][last_pred_idx])

    last_preds = np.array(last_preds)
    true_labels = np.array(true_labels)
    #print(f"true: {true_labels}")
    #print(f"pred: {last_preds}")
    accuracy = accuracy_score(true_labels, last_preds)
    report = classification_report(true_labels, last_preds, output_dict=True, zero_division=0)
    
    return {
        "accuracy": round(accuracy, 4),
        "precision": round(report["weighted avg"]["precision"], 4),
        "recall": round(report["weighted avg"]["recall"], 4),
        "f1": round(report["weighted avg"]["f1-score"], 4),
        "hallucination_rate": round(hallucination_rate(last_preds,valid_labels),4),
        "matthews_corrcoef": round(matthews_corrcoef(last_preds,true_labels),4),
        "cohen_kappa_score": round(cohen_kappa_score(last_preds,true_labels),4),
    }

def custom_compute_metrics(eval_pred, valid_labels, tokenizer, pad_token_id = -100):
    preds, labels = eval_pred
    
    
    if isinstance(preds, tuple):
        preds = preds[0]

    #preds = np.where(logits != -100, logits, tokenizer.pad_token_id)

    preds = [
        preds[i][
            np.where(
                np.logical_and(
                    preds[i] != pad_token_id,
                    #preds[i] != 271,
                    #preds[i] != 512,
                    preds[i] != -100,
                )
            )[0][-1]
        ] for i in range(preds.shape[0])
    ]

    preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    
    cleaned_preds = []
    for p in preds:
        try:
            cleaned_preds.append(int(p.strip()))
        except:
            cleaned_preds.append(-1)
    ## DEBUG
    #print(preds)
    #print(labels)
    ##

    accuracy = accuracy_score(labels, cleaned_preds)
    report = classification_report(labels, cleaned_preds, output_dict=True, zero_division=0)
    
    out_dict = {
        "accuracy": round(accuracy, 4),
        "precision": round(report["weighted avg"]["precision"], 4),
        "recall": round(report["weighted avg"]["recall"], 4),
        "f1": round(report["weighted avg"]["f1-score"], 4),
        "hallucination_rate": round(hallucination_rate(cleaned_preds, valid_labels),4),
        "matthews_corrcoef": round(matthews_corrcoef(labels, cleaned_preds),4),
        "cohen_kappa_score": round(cohen_kappa_score(labels, cleaned_preds),4),
    }
    
    return out_dict

def custom_compute_cls_metrics(eval_pred, valid_labels, tokenid2label, pad_token_id = -100):
    preds, labels = eval_pred
    
    if isinstance(preds, tuple):
        preds = preds[0]

    #preds = np.where(logits != -100, logits, tokenizer.pad_token_id)

    preds = [
        preds[i][
            np.where(
                np.logical_and(
                    preds[i] != pad_token_id,
                    #preds[i] != 271,
                    #preds[i] != 512,
                    preds[i] != -100,
                )
            )[0][-1]
        ] for i in range(preds.shape[0])
    ]
    labels = [tokenid2label[label] for label in labels]
    ## DEBUG
    #print(preds)
    #print(labels)
    ##

    accuracy = accuracy_score(labels, preds)
    report = classification_report(labels, preds, output_dict=True, zero_division=0)
    
    out_dict = {
        "accuracy": round(accuracy, 4),
        "precision": round(report["weighted avg"]["precision"], 4),
        "recall": round(report["weighted avg"]["recall"], 4),
        "f1": round(report["weighted avg"]["f1-score"], 4),
        "hallucination_rate": round(hallucination_rate(preds, valid_labels),4),
        "matthews_corrcoef": round(matthews_corrcoef(labels, preds),4),
        "cohen_kappa_score": round(cohen_kappa_score(labels, preds),4),
    }

    return out_dict

#def compute_cls_metrics(pred):
#    preds = pred.predictions.argmax(-1)
#    labels = pred.label_ids
#
#    valid = labels != -100
#    correct = (preds == labels) & valid
#
#    accuracy = correct.sum() / valid.sum()
#    return {"accuracy": accuracy.item()}

  