"""
Galaxy Morphology Classification using EfficientNet (Keras / TensorFlow)

Save this file as `galaxy_efficientnet.py` and run with Python 3.8+.
Dependencies:
  - tensorflow (>=2.6)
  - pandas
  - scikit-learn

What it does:
  - Loads `training_solutions_rev1.csv` and images from `images_training_rev1` (expected in same folder or give paths).
  - Builds an EfficientNetB0-based model with a sigmoid output for multi-label predictions.
  - Trains the model and saves the best checkpoint.
  - Provides a utility function `evaluate_on_full_dataset()` which prints accuracy on the full dataset once.

Notes / assumptions:
  - `training_solutions_rev1.csv` should have a column with the image id (called e.g. 'GalaxyID') and the label columns.
  - Label columns are inferred automatically as every column except the first id column.
  - This is implemented as multi-label binary classification (one sigmoid per target). We use `binary_crossentropy`.

"""

import os
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

# ------------------------------- Configuration -------------------------------
DEFAULT_IMG_DIR = "images_training_rev1"
DEFAULT_CSV_PATH = "training_solutions_rev1.csv"
IMG_SIZE = (224, 224)   # EfficientNetB0 default
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE

# ------------------------------- Utilities ----------------------------------

def infer_label_columns(df: pd.DataFrame):
    """Return label column names by removing the first column (assumed ID).
    If the CSV already has a column explicitly called 'GalaxyID' or 'ID', it will be removed.
    """
    cols = list(df.columns)
    # Try to find ID-like column name
    id_candidates = [c for c in cols if c.lower() in ("galaxyid", "id", "imageid", "image_id")] or len(cols) > 0 and cols[0]
    # Prefer explicit candidate
    explicit_id = None
    for cand in cols:
        if cand.lower() in ("galaxyid", "id", "imageid", "image_id"):
            explicit_id = cand
            break
    if explicit_id is None:
        # fallback: first column
        explicit_id = cols[0]
    label_cols = [c for c in cols if c != explicit_id]
    return explicit_id, label_cols


def load_dataframe(csv_path: str, images_dir: str):
    df = pd.read_csv(csv_path)
    id_col, label_cols = infer_label_columns(df)
    # build file path column (GalaxyID.jpg)
    df = df.copy()
    df["filepath"] = df[id_col].astype(str) + ".jpg"
    df["filepath"] = df["filepath"].apply(lambda x: os.path.join(images_dir, x))
    # Ensure labels are floats (0/1 or probabilities). We'll cast to float32.
    df[label_cols] = df[label_cols].astype(float)
    return df, id_col, label_cols


def make_dataset_from_df(df: pd.DataFrame, label_cols, batch_size=BATCH_SIZE, is_training=False):
    filepaths = df['filepath'].values
    labels = df[label_cols].values.astype(np.float32)
    num_labels = labels.shape[1]

    ds = tf.data.Dataset.from_tensor_slices((filepaths, labels))

    if is_training:
        ds = ds.shuffle(buffer_size=len(filepaths))

    def _process_path(path, label):
        # path is a scalar string tensor
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize(img, IMG_SIZE)
        img = tf.keras.applications.efficientnet.preprocess_input(img * 255.0)

        if is_training:
            img = tf.image.random_flip_left_right(img)
            img = tf.image.random_flip_up_down(img)
            img = tf.image.random_brightness(img, max_delta=0.1)
            # rotation if tfa available
            if 'tfa' in globals():
                angle = tf.random.uniform([], minval=-0.1, maxval=0.1)
                img = tfa.image.rotate(img, angles=angle)

        # ensure known shapes for Keras
        img = tf.ensure_shape(img, [IMG_SIZE[0], IMG_SIZE[1], 3])
        label = tf.ensure_shape(label, [num_labels])
        return img, label

    ds = ds.map(_process_path, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds

# ------------------------------- Model --------------------------------------

def build_model(num_labels, input_shape=(*IMG_SIZE, 3), dropout_rate=0.4):
    inputs = tf.keras.Input(shape=input_shape)
    #base = tf.keras.applications.EfficientNetB0(include_top=False, weights='imagenet', input_tensor=inputs, pooling='avg')
    base = tf.keras.applications.EfficientNetB0(include_top=False, weights=None, input_tensor=inputs,
                                                pooling='avg')
    x = base.output
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    outputs = tf.keras.layers.Dense(num_labels, activation='sigmoid')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

# ------------------------------- Evaluation util ----------------------------

def evaluate_on_full_dataset(model, dataset, verbose=True, threshold=0.5):
    """
    Run model.predict on full dataset once and print accuracy statistics.

    Accuracy definitions printed:
      - label_accuracy: fraction of individual labels predicted correctly (after thresholding)
      - sample_exact_match: fraction of samples where all labels exactly match the thresholded ground truth

    Returns a dict with computed metrics.
    """
    y_true_list = []
    y_pred_list = []
    for x_batch, y_batch in dataset:
        preds = model.predict(x_batch, verbose=0)
        y_true_list.append(y_batch.numpy())
        y_pred_list.append(preds)
    y_true = np.vstack(y_true_list)
    y_pred = np.vstack(y_pred_list)

    y_pred_bin = (y_pred >= threshold).astype(np.float32)
    # If ground truth values are probabilities, round them
    y_true_bin = (y_true >= 0.5).astype(np.float32)

    total_labels = np.prod(y_true_bin.shape)
    correct_labels = (y_pred_bin == y_true_bin).sum()
    label_accuracy = correct_labels / total_labels

    exact_matches = np.all(y_pred_bin == y_true_bin, axis=1).sum()
    sample_exact_match = exact_matches / y_true_bin.shape[0]

    if verbose:
        print(f"Full dataset label-wise accuracy: {label_accuracy:.4f}")
        print(f"Full dataset sample exact-match accuracy: {sample_exact_match:.4f}")
        print(f"Total samples evaluated: {y_true_bin.shape[0]}")
        print(f"Total individual labels evaluated: {total_labels}")

    return {
        'label_accuracy': float(label_accuracy),
        'sample_exact_match': float(sample_exact_match),
        'n_samples': int(y_true_bin.shape[0]),
        'n_labels': int(total_labels),
    }

# ------------------------------- Training -----------------------------------

def train(args):
    # Load dataframe
    df, id_col, label_cols = load_dataframe(args.csv, args.img_dir)
    print(f"Found {len(df)} examples and {len(label_cols)} label columns. ID column: {id_col}")

    train_df, val_df = train_test_split(df, test_size=args.val_split, random_state=42)

    # build datasets
    try:
        import tensorflow_addons as tfa
        globals()['tfa'] = tfa
        print("tensorflow_addons available: rotation augmentations enabled")
    except Exception:
        print("tensorflow_addons not available: skipping rotation augmentation")

    train_ds = make_dataset_from_df(train_df, label_cols, batch_size=args.batch_size, is_training=True)
    val_ds = make_dataset_from_df(val_df, label_cols, batch_size=args.batch_size, is_training=False)

    model = build_model(num_labels=len(label_cols))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss='binary_crossentropy',
        metrics=[tf.keras.metrics.BinaryAccuracy(name='binary_accuracy'), tf.keras.metrics.AUC(name='auc')]
    )
    model.summary()

    # callbacks
    ckpt_path = args.checkpoint or 'best_galaxy_efficientnet.h5'
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(ckpt_path, monitor='val_auc', mode='max', save_best_only=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor='val_auc', patience=8, mode='max', restore_best_weights=True, verbose=1)
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    # Evaluate on the full dataset (train+val or user-specified) once and print metrics
    if args.evaluate_full:
        print("\nEvaluating model on the full dataset (train + val)...")
        full_df = pd.concat([train_df, val_df])
        full_ds = make_dataset_from_df(full_df, label_cols, batch_size=args.batch_size, is_training=False)
        evaluate_on_full_dataset(model, full_ds)

    # Save final model
    if args.save_model:
        out_path = args.save_model
        model.save(out_path)
        print(f"Saved final model to {out_path}")

    return model, history

# ------------------------------- CLI ----------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Galaxy Morphology Classification with EfficientNetB0')
    parser.add_argument('--img_dir', type=str, default=DEFAULT_IMG_DIR, help='folder with training images')
    parser.add_argument('--csv', type=str, default=DEFAULT_CSV_PATH, help='path to training_solutions_rev1.csv')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--val_split', type=float, default=0.1)
    parser.add_argument('--checkpoint', type=str, default=None, help='where to save best checkpoint (.h5)')
    parser.add_argument('--save_model', type=str, default='galaxy_efficientnet_final.h5')
    parser.add_argument('--evaluate_full', action='store_true', help='evaluate on the full dataset and print accuracy once')

    args = parser.parse_args()
    train(args)
