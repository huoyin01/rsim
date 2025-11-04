import argparse
import os
import glob

import datasets
import numpy as np
from huggingface_hub import HfApi


datasets.logging.set_verbosity(datasets.logging.INFO)
logger = datasets.logging.get_logger(__name__)

root = os.path.join('..', 'data', 'new4_fix')


def parse_args():
    parser = argparse.ArgumentParser(description="Upload dataset to HuggingFace Hub")
    parser.add_argument('--repo_name', type=str, required=True, help="Name of the repository")
    parser.add_argument('--hub_token', type=str, required=True, help="HuggingFace Hub token")
    parser.add_argument('--version', type=str, required=True, help="Version of the dataset")
    parser.add_argument('--private', action='store_true', help="Whether to make the repository private")
    args = parser.parse_args()
    return args
    
def get_version():
    args = parse_args()
    return args.version

class DataBuilder(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name='rsim',
            version=datasets.Version(get_version()),
            description='rsim dataset'),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features({
                'instruction': datasets.Sequence(datasets.Value('int32')),
                'coverage_points': datasets.Value('int32'),
                'coverage_modules': datasets.Sequence(datasets.Value('int32')),
            })
        )

    def _split_generators(self, dl_manager):
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={'root': os.path.join(root, 'train')}),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={'root': os.path.join(root, 'test')}),
        ]

    def _generate_examples(self, root):
        eid = 0
        for path in glob.glob(os.path.join(root, '*')):
            logger.info(f'Generating examples from {path!r}...')
            with open(path, 'r', encoding='utf-8') as f:
                last_coverage_line = f.readline()
                parts = last_coverage_line.strip().split(',')
                last_coverage_points = int(parts[-23])
                last_coverage_modules = np.array([int(coverage_module) for coverage_module in parts[-22:]])
                for line in f:
                    parts = line.strip().split(',')
                    instruction = [int(ins) for ins in parts[:-23]]
                    
                    coverage_points = int(parts[-23]) 
                    new_coverage_points = coverage_points - last_coverage_points
                    last_coverage_points = coverage_points
                    
                    coverage_modules = np.array([int(coverage_module) for coverage_module in parts[-22:]])
                    new_coverage_modules = coverage_modules - last_coverage_modules
                    last_coverage_modules = coverage_modules
                    yield eid, {
                        'instruction': instruction,
                        'coverage_points': new_coverage_points,
                        'coverage_modules': new_coverage_modules,
                    }
                    eid += 1


def main():
    args = parse_args()
    
    builder = DataBuilder(root)
    builder.download_and_prepare(download_mode=datasets.DownloadMode.FORCE_REDOWNLOAD)
    dataset = builder.as_dataset(split='train')
    
    api = HfApi(token=args.hub_token)
    repo_id = api.create_repo(args.repo_name, repo_type="dataset", exist_ok=True, token=args.hub_token).repo_id
    dataset.push_to_hub(repo_id, private=args.private)
    print(f"Dataset {args.repo_name} has been uploaded to your HuggingFace Hub!")


if __name__ == '__main__':
    main()          
