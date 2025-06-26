import os
import datetime
import sys
import matplotlib.pyplot as plt


def save_train_result(num_data, epochs, learning_rate,loss):
    learning_rate_list = [learning_rate] * epochs

    # ===== グラフ作成 =====
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    ax[0].set_title("Learning Rate")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Learning Rate")
    ax[0].plot(range(1, epochs + 1), learning_rate_list, label=f"LR={learning_rate}")

    ax[1].set_title("Loss")
    ax[1].set_xlabel("Epoch")
    ax[1].plot(range(1, epochs + 1), loss, label="Loss")

    fig.suptitle(f"Dataset Size: {num_data}")
    plt.legend()
    plt.tight_layout()

    # ===== フォルダ階層の作成 =====
    # data100/epoch100/learningrate0.01 のような構造にする
    save_dir = os.path.join(
        "/home/ryuichi/tree/TREE_PROJ/train_result/svsae",
        f"data{num_data}",
        f"epoch{epochs}",
        f"learningrate{learning_rate}"
    )
    os.makedirs(save_dir, exist_ok=True)

    # ===== 現在時刻を含むファイル名を作成 =====
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"result_{current_time}.png"

    # ===== 保存パスを作成して保存 =====
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path)
    plt.show()

    print(f"結果を保存しました: {save_path}")

# # Example usage
# save_train_result(num_data=100, epochs=100, learning_rate=0.01)
