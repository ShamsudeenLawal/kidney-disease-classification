import os
import zipfile
import shutil
import sys
from pathlib import Path

import gdown
from sklearn.model_selection import train_test_split

from kidney_disease_classifier import logger
from kidney_disease_classifier.entity.config_entity import DataIngestionConfig

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self):
        try:
            zip_path = Path(self.config.local_data_file)

            os.makedirs(self.config.root_dir, exist_ok=True)

            if zip_path.exists():
                logger.info("Dataset already downloaded.")
                return

            logger.info("Downloading dataset...")

            file_id = self.config.source_url.split("/")[-2]
            url = f"https://drive.google.com/uc?export=download&id={file_id}"

            gdown.download(url, str(zip_path), quiet=False)

            logger.info(f"Downloaded to {zip_path}")

        except Exception as e:
            raise Exception(e, sys)
        
    def extract_zip_file(self):
        try:
            raw_dir = Path(self.config.raw_dir)
            raw_dir.mkdir(parents=True, exist_ok=True)

            dataset_marker = raw_dir / "dataset"

            if dataset_marker.exists():
                logger.info("Dataset already extracted.")
                return

            logger.info("Extracting zip file...")

            with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
                zip_ref.extractall(raw_dir)

            logger.info(f"Extracted to {raw_dir}")

        except Exception as e:
            raise Exception(e, sys)

    def rename_dataset_folder(self):
        try:
            raw_dir = Path(self.config.raw_dir)
            dataset_path = raw_dir / "dataset"

            if dataset_path.exists():
                logger.info("Dataset already renamed.")
                return

            folders = [f for f in raw_dir.iterdir() if f.is_dir()]

            if len(folders) != 1:
                raise Exception(f"Expected 1 folder, found {len(folders)}")

            folders[0].rename(dataset_path)

            logger.info(f"Renamed to {dataset_path}")

        except Exception as e:
            raise Exception(e, sys)

    def split_dataset(self, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=42):

        try:
            dataset_dir = Path(self.config.raw_dir) / "dataset"

            split_cfg = self.config.split

            train_dir = Path(split_cfg.train_dir)
            val_dir = Path(split_cfg.val_dir)
            test_dir = Path(split_cfg.test_dir)

            for d in [train_dir, val_dir, test_dir]:
                d.mkdir(parents=True, exist_ok=True)

            logger.info("Splitting dataset...")

            filepaths, labels = [], []

            for class_folder in dataset_dir.iterdir():
                if not class_folder.is_dir():
                    continue

                for file in class_folder.glob("*"):
                    filepaths.append(file)
                    labels.append(class_folder.name)

            # train + temp
            X_train, X_temp, y_train, y_temp = train_test_split(
                filepaths,
                labels,
                test_size=(1 - train_ratio),
                stratify=labels,
                random_state=random_state
            )

            val_ratio_adj = val_ratio / (val_ratio + test_ratio)

            X_val, X_test, y_val, y_test = train_test_split(
                X_temp,
                y_temp,
                test_size=(1 - val_ratio_adj),
                stratify=y_temp,
                random_state=random_state
            )

            def copy_files(files, labels, target_dir):
                for file, label in zip(files, labels):
                    class_dir = target_dir / label
                    class_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy(file, class_dir)

            copy_files(X_train, y_train, train_dir)
            copy_files(X_val, y_val, val_dir)
            copy_files(X_test, y_test, test_dir)

            logger.info("Dataset split completed successfully.")

        except Exception as e:
            raise Exception(e, sys)
        
    def run(self):
        self.download_file()
        self.extract_zip_file()
        self.rename_dataset_folder()
        self.split_dataset()


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self):
        try:
            zip_path = Path(self.config.local_data_file)
            os.makedirs(self.config.root_dir, exist_ok=True)

            if zip_path.exists():
                logger.info("Dataset already downloaded.")
                return

            logger.info("Downloading dataset...")

            file_id = self.config.source_url.split("/")[-2]
            url = f"https://drive.google.com/uc?export=download&id={file_id}"

            gdown.download(url, str(zip_path), quiet=False)

            logger.info(f"Downloaded: {zip_path}")

        except Exception as e:
            raise Exception(e, sys)

    def extract_zip_file(self):
        try:
            raw_dir = Path(self.config.raw_dir)
            raw_dir.mkdir(parents=True, exist_ok=True)

            dataset_dir = raw_dir / "dataset"

            # 🔥 safe skip: check actual dataset folder, not just files
            if dataset_dir.exists() and any(dataset_dir.rglob("*")):
                logger.info("Dataset already extracted.")
                return

            logger.info("Extracting dataset...")

            with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
                zip_ref.extractall(raw_dir)

            logger.info(f"Extraction completed: {raw_dir}")

        except Exception as e:
            raise Exception(e, sys)

    def rename_dataset_folder(self):
        try:
            raw_dir = Path(self.config.raw_dir)
            dataset_dir = raw_dir / "dataset"

            if dataset_dir.exists():
                logger.info("Dataset already renamed.")
                return

            folders = [f for f in raw_dir.iterdir() if f.is_dir()]

            if len(folders) != 1:
                raise Exception(f"Expected 1 dataset folder, found {len(folders)}")

            folders[0].rename(dataset_dir)

            logger.info(f"Dataset renamed to {dataset_dir}")

        except Exception as e:
            raise Exception(e, sys)

    def _is_split_valid(self, split_cfg: dict) -> bool:
        """
        Validate if dataset split already exists and is usable
        """

        train = Path(split_cfg["train_dir"])
        val = Path(split_cfg["val_dir"])
        test = Path(split_cfg["test_dir"])

        # must exist
        if not (train.exists() and val.exists() and test.exists()):
            return False

        # must contain data
        if len(list(train.rglob("*"))) == 0:
            return False
        if len(list(val.rglob("*"))) == 0:
            return False
        if len(list(test.rglob("*"))) == 0:
            return False

        return True

    def split_dataset(self, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=42):

        try:
            split_cfg = self.config.split

            # 🔥 SAFE SKIP CONDITION
            if self._is_split_valid(split_cfg):
                logger.info("Valid split already exists. Skipping split.")
                return

            dataset_dir = Path(self.config.raw_dir) / "dataset"

            train_dir = Path(split_cfg.train_dir)
            val_dir = Path(split_cfg.val_dir)
            test_dir = Path(split_cfg.test_dir)

            for d in [train_dir, val_dir, test_dir]:
                d.mkdir(parents=True, exist_ok=True)

            logger.info("Creating dataset split...")

            filepaths, labels = [], []

            for class_folder in dataset_dir.iterdir():
                if not class_folder.is_dir():
                    continue

                for file in class_folder.glob("*"):
                    filepaths.append(file)
                    labels.append(class_folder.name)

            # train / temp split
            X_train, X_temp, y_train, y_temp = train_test_split(
                filepaths,
                labels,
                test_size=(1 - train_ratio),
                stratify=labels,
                random_state=random_state
            )

            val_adj = val_ratio / (val_ratio + test_ratio)

            X_val, X_test, y_val, y_test = train_test_split(
                X_temp,
                y_temp,
                test_size=(1 - val_adj),
                stratify=y_temp,
                random_state=random_state
            )

            def copy_files(files, labels, target_dir):
                for file, label in zip(files, labels):
                    class_dir = target_dir / label
                    class_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy(file, class_dir)

            copy_files(X_train, y_train, train_dir)
            copy_files(X_val, y_val, val_dir)
            copy_files(X_test, y_test, test_dir)

            logger.info("Dataset split completed.")

        except Exception as e:
            raise Exception(e, sys)

    def run(self):
        self.download_file()
        self.extract_zip_file()
        self.rename_dataset_folder()
        self.split_dataset()