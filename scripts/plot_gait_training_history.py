"""Plot recorded optimization and validation history without reading test data."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot_history(training_directory, output_directory, *, title):
    source, destination = Path(training_directory), Path(output_directory)
    names = ('training_history.json', 'resolved_training.json', 'training_result.json')
    contents = {name: (source / name).read_bytes() for name in names}
    history, config, result = [json.loads(contents[name]) for name in names]
    if not history:
        raise ValueError('No recorded optimization epochs to plot')
    epochs = [row['epoch'] for row in history]
    if epochs != sorted(set(epochs)):
        raise ValueError('Epochs must be unique and increasing')
    selection = config['selection']
    label = {'validation_macro_average_precision': 'Validation macro average precision',
             'validation_macro_normalized_mae': 'Validation mean standardized MAE'}.get(selection['name'])
    if label is None:
        raise ValueError('Unknown predeclared validation selection criterion')
    if result['selection'] != selection:
        raise ValueError('Training result and declared selection criterion disagree')
    validation = {row['epoch']: row['validation']['selection_score']
                  for row in history if row.get('validation') is not None}
    best_epoch = result['best_epoch']
    if best_epoch is not None and (best_epoch not in validation or
                                  validation[best_epoch] != result['best_validation_score']):
        raise ValueError('Selected checkpoint is not supported by recorded validation')
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.5))
    colors = {'total': '#205e5a', 'classification': '#ae6c31',
              'regression': '#4f75a8', 'evaluability': '#8c679b'}
    for task, color in colors.items():
        values = [row['training_loss'].get(task) for row in history]
        observed = [value for value in values if value is not None]
        if not observed:
            continue
        if not np.isfinite(observed).all():
            plt.close(fig)
            raise ValueError('Nonfinite recorded training loss')
        axes[0].plot(epochs, [np.nan if value is None else value for value in values],
                     label=task.capitalize(), color=color, linewidth=1.6, marker='.', markersize=4)
    axes[0].set(xlabel='Epoch', ylabel='Mean loss over optimized batches', title='Training objective')
    axes[0].legend(frameon=False)
    if validation:
        if not np.isfinite(list(validation.values())).all():
            plt.close(fig)
            raise ValueError('Nonfinite recorded validation score')
        axes[1].plot(list(validation), list(validation.values()), color='#205e5a', marker='o', markersize=3)
    else:
        axes[1].text(.5, .5, 'No complete validation pass', transform=axes[1].transAxes,
                     ha='center', va='center')
    if best_epoch is not None:
        axes[1].scatter([best_epoch], [validation[best_epoch]], s=70, facecolors='none',
                        edgecolors='#ae6c31', linewidths=2, label=f'Selected epoch {best_epoch}')
        axes[1].legend(frameon=False)
    axes[1].set(xlabel='Epoch', ylabel=label,
                title='Validation selection · ' + ('higher is better' if selection['direction'] == 'maximize' else 'lower is better'))
    if selection['direction'] == 'maximize':
        axes[1].set_ylim(-.03, 1.03)
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=.18)
        if len(epochs) == 1:
            ax.set_xlim(epochs[0] - .5, epochs[0] + .5)
        ax.xaxis.get_major_locator().set_params(integer=True)
    fig.suptitle(title, fontsize=13)
    fig.text(.5, .02, 'Selection uses validation only. Missing validation checks are omitted; held-out test data is not used.\n'
             'Loss weights: classification 1.0 · standardized numeric Huber 0.5 · evaluability 0.2.',
             ha='center', fontsize=8, color='#444444')
    fig.tight_layout(rect=(0, .09, 1, .95))
    destination.mkdir(parents=True, exist_ok=True)
    paths = []
    for extension in ('svg', 'png'):
        path = destination / ('training_curves.' + extension)
        temporary = path.with_name(path.stem + '.tmp.' + extension)
        fig.savefig(temporary, dpi=160, facecolor='white')
        temporary.replace(path)
        paths.append(path)
    plt.close(fig)
    record = dict(generated_at=datetime.now(timezone.utc).isoformat(), title=title,
                  training_directory=str(source.resolve()), epoch_count=len(epochs),
                  complete_validation_checks=len(validation), selected_epoch=best_epoch,
                  selection=selection, input_provenance=config['input_provenance'],
                  inputs_sha256={name: hashlib.sha256(blob).hexdigest() for name, blob in contents.items()},
                  generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  figures={path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
                  held_out_test_consumed=False)
    path = destination / 'training_curves.json'
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('training_directory', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--title', required=True)
    args = parser.parse_args()
    record = plot_history(args.training_directory, args.output, title=args.title)
    print(json.dumps({k: record[k] for k in ['epoch_count', 'complete_validation_checks', 'selected_epoch', 'held_out_test_consumed']}))
