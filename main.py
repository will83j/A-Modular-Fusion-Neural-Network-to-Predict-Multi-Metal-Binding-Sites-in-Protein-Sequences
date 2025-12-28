import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Conv1D, Dropout, Dense, TimeDistributed
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import precision_recall_fscore_support
import tensorflow as tf
import tensorflow.keras.backend as K
import pickle
import gc

# Reading data
df = pd.read_csv('metal_new.csv')
seqs = df['sequence'].astype(str).tolist()
sites = df['site'].apply(lambda s: [int(p) for p in eval(str(s))] if isinstance(s, (str, bytes)) or not pd.isnull(s) else []).tolist()
metals = df['metal'].apply(lambda s: eval(str(s)) if isinstance(s, (str, bytes)) or not pd.isnull(s) else []).tolist()


# Defining metal classes
metal_types = ['Zn', 'Fe', 'Mg']
metal_to_index = {m: i for i, m in enumerate(metal_types)}
MAX_LEN = 500
N_FOLDS = 5

# Tokenizing
tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(seqs)
seqs_int = tokenizer.texts_to_sequences(seqs)
X = pad_sequences(seqs_int, maxlen=MAX_LEN, padding='post', truncating='post')
vocab_size = len(tokenizer.word_index) + 1
embed_dim = 64

# Saving tokenizer
with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

# Creating multi-metal labelling matrix (for dividing dataset)
y_all = np.zeros((len(seqs), MAX_LEN, len(metal_types)), dtype='float32')
for i, pos_list in enumerate(sites):
    for j, p in enumerate(pos_list):
        if p < MAX_LEN:
            metal_name = metals[i][j]
            if metal_name in metal_to_index:
                y_all[i, p - 1, metal_to_index[metal_name]] = 1.0


sample_labels = np.any(y_all, axis=1)
stratify_labels = np.argmax(sample_labels, axis=1)

# Dividing dataset - 15% for testing, 85% for training
X_temp, X_test, y_temp, y_test, sites_temp, sites_test, metals_temp, metals_test, stratify_temp, stratify_test = train_test_split(
    X, y_all, sites, metals, stratify_labels, test_size=0.15, random_state=42, stratify=stratify_labels
)


# Labelling metal binding residues
def build_labels_for_metal(target_metal, sites_data, metals_data, data_size):
    y = np.zeros((data_size, MAX_LEN, 1), dtype='float32')
    for i, pos_list in enumerate(sites_data):
        for j, p in enumerate(pos_list):
            if p < MAX_LEN and metals_data[i][j] == target_metal:
                y[i, p - 1, 0] = 1.0
    return y

# Counting positive binding sites for each metal
def count_metal_samples(sites_data, metals_data):
    counts = {m: 0 for m in metal_types}
    for i, pos_list in enumerate(sites_data):
        for j, p in enumerate(pos_list):
            m = metals_data[i][j]
            if m in counts:
                counts[m] += 1
    return counts

metal_positive_counts = count_metal_samples(sites_temp, metals_temp)

# Defining weighted binary cross entropy
def weighted_binary_crossentropy(pos_weight):
    def loss(y_true, y_pred):
        loss = K.binary_crossentropy(y_true, y_pred)
        weight = tf.where(K.equal(y_true, 1), pos_weight, 1.0)
        return K.mean(loss * weight)
    return loss

# Defining single metal model
def train_single_metal_model(X_train, X_val, y_train, y_val, pos_count, fold_id):
    total = y_train.shape[0] * y_train.shape[1]
    neg_count = total - pos_count
    pos_weight = neg_count / (pos_count + 1e-6)

    inputs = Input(shape=(MAX_LEN,), dtype='int32')
    x = Embedding(input_dim=vocab_size, output_dim=embed_dim)(inputs)
    x = Conv1D(512, 15, activation='relu', padding='same')(x)
    x = Conv1D(256, 7, activation='relu', padding='same')(x)
    x = Conv1D(128, 5, activation='relu', padding='same')(x)
    x = Conv1D(64, 3, activation='relu', padding='same')(x)
    x = Dropout(0.3)(x)
    outputs = TimeDistributed(Dense(1, activation='sigmoid'))(x)

    model = Model(inputs, outputs)
    model.compile(optimizer='adam',
                 loss=weighted_binary_crossentropy(pos_weight),
                 metrics=['accuracy'])

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=20,
        batch_size=128,
        callbacks=[EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)],
        verbose=0
    )
    return model

# Defining fusion model
def train_fusion_model(metal_preds_train, metal_preds_val, y_train_all, y_val_all, fold_id):
    fusion_in = Input(shape=(MAX_LEN, len(metal_types)))
    x = Dense(256, activation='relu')(fusion_in)
    x = Dropout(0.2)(x)
    fusion_out = Dense(len(metal_types), activation='sigmoid')(x)

    fusion_model = Model(fusion_in, fusion_out)
    fusion_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    fusion_model.fit(
        metal_preds_train, y_train_all,
        validation_data=(metal_preds_val, y_val_all),
        epochs=30,
        batch_size=128,
        callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)],
        verbose=0
    )
    return fusion_model

# Getting model prediction for protein
def get_metal_preds_on_X(X, metal_models):
    preds = []
    for m in metal_types:
        pred = metal_models[m].predict(X, batch_size=64, verbose=0)
        preds.append(pred.squeeze(-1))
    return np.stack(preds, axis=-1)

# Five-fold cross validation
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
cv_results = []
all_fold_models = {'metal': [], 'fusion': []}

for fold, (train_idx, val_idx) in enumerate(skf.split(X_temp, stratify_temp)):
    print(f"\n--- Fold {fold+1}/{N_FOLDS} ---")

    # Training and validation dataset for current fold
    X_train, X_val = X_temp[train_idx], X_temp[val_idx]
    y_train_all, y_val_all = y_temp[train_idx], y_temp[val_idx]
    sites_train = [sites_temp[i] for i in train_idx]
    sites_val = [sites_temp[i] for i in val_idx]
    metals_train = [metals_temp[i] for i in train_idx]
    metals_val = [metals_temp[i] for i in val_idx]

    # Positive binding sites for current fold
    fold_metal_counts = count_metal_samples(sites_train, metals_train)

    # Training single metal model
    fold_metal_models = {}
    for m in metal_types:
        y_train_m = build_labels_for_metal(m, sites_train, metals_train, len(X_train))
        y_val_m = build_labels_for_metal(m, sites_val, metals_val, len(X_val))

        pos_count = fold_metal_counts[m]
        model = train_single_metal_model(X_train, X_val, y_train_m, y_val_m, pos_count, fold)
        fold_metal_models[m] = model

    # Getting predictions from single metal model
    metal_preds_train = get_metal_preds_on_X(X_train, fold_metal_models)
    metal_preds_val = get_metal_preds_on_X(X_val, fold_metal_models)

    # Training fusion model
    fold_fusion_model = train_fusion_model(metal_preds_train, metal_preds_val,
                                         y_train_all, y_val_all, fold)

    # Evaluate using validation set
    y_pred_prob = fold_fusion_model.predict(metal_preds_val, verbose=0)
    y_pred = (y_pred_prob >= 0.45).astype(int)

    # Calculating evaluation matrics
    y_val_flat = y_val_all.reshape(-1, len(metal_types))
    y_pred_flat = y_pred.reshape(-1, len(metal_types))

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_val_flat, y_pred_flat, average='macro', zero_division=0
    )

    cv_results.append({
        'fold': fold + 1,
        'precision': precision,
        'recall': recall,
        'f1': f1
    })

    print(f"Fold {fold+1} - Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")

    # Save model
    all_fold_models['metal'].append(fold_metal_models)
    all_fold_models['fusion'].append(fold_fusion_model)

    del metal_preds_train, metal_preds_val, y_pred_prob
    gc.collect()

# Summary of cross validation matrics
print("Summary of cross validation results")
cv_precision = [r['precision'] for r in cv_results]
cv_recall = [r['recall'] for r in cv_results]
cv_f1 = [r['f1'] for r in cv_results]

print(f"Average of cross validation results:")
print(f"Precision: {np.mean(cv_precision):.3f} ± {np.std(cv_precision):.3f}")
print(f"Recall: {np.mean(cv_recall):.3f} ± {np.std(cv_recall):.3f}")
print(f"F1 Score: {np.mean(cv_f1):.3f} ± {np.std(cv_f1):.3f}")

# Test using testing dataset
test_predictions = []
for fold in range(N_FOLDS):
    metal_preds_test = get_metal_preds_on_X(X_test, all_fold_models['metal'][fold])
    fold_pred = all_fold_models['fusion'][fold].predict(metal_preds_test, verbose=0)
    test_predictions.append(fold_pred)

ensemble_pred_prob = np.mean(test_predictions, axis=0)

# Evaluating different thresholds
thresholds = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

from sklearn.metrics import matthews_corrcoef, roc_auc_score, average_precision_score

for threshold in thresholds:
    y_pred_test = (ensemble_pred_prob >= threshold).astype(int)
    y_test_flat = y_test.reshape(-1, len(metal_types))
    y_pred_test_flat = y_pred_test.reshape(-1, len(metal_types))
    y_prob_flat = ensemble_pred_prob.reshape(-1, len(metal_types))

    # Evaluation matrics for each metal
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test_flat, y_pred_test_flat, average=None, zero_division=0
    )
    for i, metal in enumerate(metal_types):
        print(f"{metal}: P={precision[i]:.3f}, R={recall[i]:.3f}, F1={f1[i]:.3f}")

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_test_flat, y_pred_test_flat, average='macro', zero_division=0
    )
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        y_test_flat, y_pred_test_flat, average='micro', zero_division=0
    )
    print(f"Macro: P={precision_macro:.3f}, R={recall_macro:.3f}, F1={f1_macro:.3f}")
    print(f"Micro: P={precision_micro:.3f}, R={recall_micro:.3f}, F1={f1_micro:.3f}")

    # MCC, AUC, AUPRC
    y_prob_flat = ensemble_pred_prob.reshape(-1, len(metal_types))
    # MCC
    mcc = matthews_corrcoef(y_test_flat.flatten(), y_pred_test_flat.flatten())
    # AUC
    try:
        auc = roc_auc_score(y_test_flat, y_prob_flat, average='micro')
    except ValueError:
        auc = float('nan')
    # AUPRC
    try:
        auprc = average_precision_score(y_test_flat, y_prob_flat, average='macro')
    except ValueError:
        auprc = float('nan')

    print(f"MCC: {mcc:.3f}")
    print(f"AUC (micro-average): {auc:.3f}")
    print(f"AUPRC (macro-average): {auprc:.3f}")

# Saving best model(based on F1)
best_fold = np.argmax(cv_f1)

# Saving model of the best fold
for m in metal_types:
    all_fold_models['metal'][best_fold][m].save(f'best_metal_model_{m}.h5')
all_fold_models['fusion'][best_fold].save('best_fusion_model.h5')

# Predicting
def predict_binding_sites_ensemble(seq, threshold=0.5, top_k=3):
    seq_int = tokenizer.texts_to_sequences([seq])
    seq_pad = pad_sequences(seq_int, maxlen=MAX_LEN, padding='post', truncating='post')

    fold_predictions = []

    for fold in range(N_FOLDS):
        metal_preds = []
        for m in metal_types:
            pred = all_fold_models['metal'][fold][m].predict(seq_pad, verbose=0)[0, :, 0]
            metal_preds.append(pred)
        metal_preds = np.stack(metal_preds, axis=-1)

        fusion_pred = all_fold_models['fusion'][fold].predict(metal_preds[np.newaxis, ...], verbose=0)[0]
        fold_predictions.append(fusion_pred)

    ensemble_pred = np.mean(fold_predictions, axis=0)

    results = []
    for i, metal_probs in enumerate(ensemble_pred):
        if np.max(metal_probs) >= threshold:
            sorted_indices = np.argsort(metal_probs)[::-1]
            top_metals = [(metal_types[j], metal_probs[j]) for j in sorted_indices[:top_k]]
            results.append((i + 1, top_metals))

    return results

# Example
test_seq = "NELRCGCPDCHCKVDPERVFNHDGEAYCSQACAEQHPNGEPCPAPDCHCERSGKVGGRDITNNQLDEALEETFPASDPISP"
predicted_sites = predict_binding_sites_ensemble(test_seq, threshold=0.3)

print("Predicted binding sites and probability")
for pos, metals in predicted_sites:
    print(f"Sites {pos}:")
    for metal, prob in metals:
        print(f"  {metal}: {prob:.3f}")



