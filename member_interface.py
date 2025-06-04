import torch
import torch.nn.functional as F
import numpy as np
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from mnist import MNIST, main as train_target_model


# ---------------------------------------------------------------
# 1. Utility: Load (or train, if not found) the target MNIST model.
# ---------------------------------------------------------------
def load_or_train_model(device, model_path='target_model.pth'):
    """
    Loads a pre‐trained MNIST model from 'model_path' if it exists;
    otherwise, invokes the existing train_target_model() function
    (which trains and saves its weights to 'target_model.pth').
    Returns the model in evaluation mode on the specified 'device'.
    """
    model = MNIST().to(device)

    if os.path.exists(model_path):
        # Log that we are loading an existing model
        print(f"[INFO] Loading target model weights from '{model_path}'.")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        # Log that the model file was not found, so training begins
        print(f"[INFO] '{model_path}' not found. Training a new target model on MNIST...")
        train_target_model()  # this will train and save 'target_model.pth'
        print(f"[INFO] Completed training. Loading saved weights from '{model_path}'.")
        model.load_state_dict(torch.load(model_path, map_location=device))

    model.eval()  # set to evaluation mode (disables dropout, batchnorm updates, etc.)
    return model


# -------------------------------------------------------------------
# 2. Utility: Given a model and DataLoader, compute each sample's
#    maximum softmax confidence for its predicted class.
# -------------------------------------------------------------------
def get_confidences(model, loader, device):
    """
    Iterates over all (input, _) pairs in 'loader'; for each input batch,
    computes the softmax over logits, then takes the maximum probability
    per example. Returns a NumPy array of shape (num_samples,)
    containing these max confidence scores.
    """
    confidences = []

    # Disable gradient computation for speed
    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(loader):
            data = data.to(device)
            logits = model(data)  # [batch_size, num_classes]
            probs = F.softmax(logits, dim=1)  # convert logits → probabilities
            max_probs, _ = probs.max(dim=1)  # select maximum probability per sample

            # Convert to CPU NumPy and append to list
            batch_conf = max_probs.cpu().numpy()
            confidences.extend(batch_conf)

            # Log progress every few batches
            if (batch_idx + 1) % 10 == 0:
                print(f"[DEBUG] Processed {(batch_idx + 1) * loader.batch_size} samples for confidence stats.")

    confidences = np.array(confidences)
    print(f"[INFO] Computed confidences for {len(confidences)} samples.")
    return confidences


# -------------------------------------------------------------
# 3. Main Function: Perform the membership inference attack
# -------------------------------------------------------------
def membership_inference_attack():
    # Determine device: prefer GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Define the same normalization used during MNIST training
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # ----------------------------------------------------------------------
    # 3a. Load MNIST datasets without shuffling, so that order is deterministic
    # ----------------------------------------------------------------------
    print("[INFO] Loading MNIST training set (members) and test set (non-members)...")
    train_ds = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    batch_size = 1000
    # note: num_workers=2 can be adjusted based on system capabilities
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    # Log dataset sizes
    print(f"[INFO] Number of training samples (members): {len(train_ds)}")
    print(f"[INFO] Number of test samples (non-members): {len(test_ds)}")

    # -------------------------------------------------------
    # 3b. Load (or train) the target MNIST classifier
    # -------------------------------------------------------
    model = load_or_train_model(device)  # logs appear inside this call

    # -----------------------------------------
    # 3c. Compute confidences for all samples
    # -----------------------------------------
    print("[INFO] Computing confidences for training set (members)...")
    confidences_train = get_confidences(model, train_loader, device)

    print("[INFO] Computing confidences for test set (non-members)...")
    confidences_test = get_confidences(model, test_loader, device)

    # -------------------------------------------------------
    # 3d. Choose threshold based on mean confidences
    # -------------------------------------------------------
    mean_train_conf = confidences_train.mean()
    mean_test_conf = confidences_test.mean()
    threshold = (mean_train_conf + mean_test_conf) / 2.0

    # Log the computed means and threshold
    print(f"[INFO] Mean confidence on training (members): {mean_train_conf:.4f}")
    print(f"[INFO] Mean confidence on test (non-members): {mean_test_conf:.4f}")
    print(f"[INFO] Selected threshold = (mean_train + mean_test) / 2 = {threshold:.4f}")

    # ---------------------------------------------------
    # 3e. Classify each sample as member / non-member
    # ---------------------------------------------------
    # True Positives (TP): member samples whose confidence > threshold
    tp_mask = confidences_train > threshold
    tp = np.sum(tp_mask)  # count of correct member predictions

    # False Negatives (FN): members with confidence <= threshold
    fn = np.sum(~tp_mask)

    # True Negatives (TN): non-member samples whose confidence <= threshold
    tn_mask = confidences_test <= threshold
    tn = np.sum(tn_mask)

    # False Positives (FP): non-members with confidence > threshold
    fp = np.sum(~tn_mask)

    # Compute rates
    tpr = tp / len(confidences_train)  # True Positive Rate (Recall on members)
    tnr = tn / len(confidences_test)  # True Negative Rate (Recall on non-members)

    # Overall attack accuracy
    total_correct = tp + tn
    total_queries = len(confidences_train) + len(confidences_test)
    attack_accuracy = total_correct / total_queries

    # ----------------------------------------------
    # 3f. Print detailed attack metrics (logging)
    # ----------------------------------------------
    print("\n=== Membership Inference Attack Metrics ===")
    print(f"Threshold used: {threshold:.4f}\n")

    print(f"[RESULT] True Positives  (members correctly detected): {tp} / {len(confidences_train)} "
          f"→ TPR = {tpr * 100:.2f}%")
    print(f"[RESULT] False Negatives (members misclassified as non-members): {fn} / {len(confidences_train)} "
          f"→ FNR = {fn / len(confidences_train) * 100:.2f}%\n")

    print(f"[RESULT] True Negatives  (non-members correctly detected): {tn} / {len(confidences_test)} "
          f"→ TNR = {tnr * 100:.2f}%")
    print(f"[RESULT] False Positives (non-members misclassified as members): {fp} / {len(confidences_test)} "
          f"→ FPR = {fp / len(confidences_test) * 100:.2f}%\n")

    print(f"[RESULT] Overall Attack Accuracy: {total_correct} / {total_queries} "
          f"→ {attack_accuracy * 100:.2f}%\n")

    # (Optional) Additional logging: Confusion matrix components
    print("[DEBUG] Confusion matrix (on combined dataset):")
    print(f"          Actual Member  |  Actual Non-member")
    print(f"Pred=Mem      {tp:5d}             {fp:5d}")
    print(f"Pred=Non-mem  {fn:5d}             {tn:5d}")
    print("=============================================\n")


if __name__ == "__main__":
    membership_inference_attack()









