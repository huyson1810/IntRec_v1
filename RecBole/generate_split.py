import os
import torch
from recbole.config import Config
from recbole.data import create_dataset, data_preparation

def main():
    # 3. Define configuration dictionary
    config_dict = {
        'dataset': 'ml-100k',
        'data_path': '/content/drive/MyDrive/Colab_Notebooks/CIKMv1/RecBole/dataset',
        'eval_args': {
            'split': {'RS': [0.8, 0.1, 0.1]},
            'order': 'RO',
            'group_by': 'user',
            'mode': 'full'
        },
        'load_col': {'inter': ['user_id', 'item_id', 'rating', 'timestamp']}
    }

    # 4. Initialize Config object using BPR as model to allow negative sampling
    config = Config(model='BPR', dataset='ml-100k', config_dict=config_dict)

    # 5. Load raw data
    dataset = create_dataset(config)
    print(dataset)

    # 6. Perform split
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # 7. Export split interactions
    save_dir = '/content/drive/MyDrive/Colab_Notebooks/CIKMv1/RecBole/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Save interaction objects
    torch.save(train_data, os.path.join(save_dir, 'ml-100k-train.pth'))
    torch.save(valid_data, os.path.join(save_dir, 'ml-100k-valid.pth'))
    torch.save(test_data, os.path.join(save_dir, 'ml-100k-test.pth'))

    print(f'Splits saved successfully to {save_dir}')

if __name__ == "__main__":
    main()
