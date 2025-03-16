# Ollama Model Manager

A set of tools for managing Ollama language models with both GUI and CLI interfaces. This toolkit allows you to list, export, import, and delete Ollama models.

## Features

- **List Models**: View all installed Ollama models with their sizes and locations
- **Export Models**: Export models to portable tar.gz archives
- **Import Models**: Import models from tar.gz archives
- **Delete Models**: Remove models to free up disk space
- **Multiple Interfaces**: Choose between a GUI (PySide6) or CLI interface

## Requirements

- Python 3.7+
- Ollama installed (https://ollama.com)
- For GUI: PySide6 (`pip install PySide6`)

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/404FoundingFather/ollama-model-manager.git
   cd ollama-model-manager
   ```

2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Command Line Interface

The command-line interface (`ollama_cli.py`) provides a simple way to manage Ollama models from the terminal.

### Usage

```
python ollama_cli.py [-h] [--verbose] {list,export,import,delete} ...
```

### Global options

- `--verbose, -v`: Enable verbose output with additional debugging information

### List Models

List all available Ollama models:

```bash
python ollama_cli.py list
```

Output as JSON:

```bash
python ollama_cli.py list --json
```

### Export Models

Export a model to a tar.gz archive:

```bash
python ollama_cli.py export <model_name> <parameter_size> <output_path>
```

Example:
```bash
python ollama_cli.py export mistral 7b ./mistral-7b.tar.gz
```

Options:
- `--force, -f`: Overwrite output file if it exists

### Import Models

Import a model from a tar.gz archive:

```bash
python ollama_cli.py import <archive_path>
```

Example:
```bash
python ollama_cli.py import ./mistral-7b.tar.gz
```

Options:
- `--name <custom_name>`: Set a custom name for the imported model (format: name:param_size)

Example with custom name:
```bash
python ollama_cli.py import ./mistral-7b.tar.gz --name "my-mistral:7b"
```

### Delete Models

Delete a model:

```bash
python ollama_cli.py delete <model_name> <parameter_size>
```

Example:
```bash
python ollama_cli.py delete mistral 7b
```

Options:
- `--force, -f`: Skip confirmation prompt


## Graphical User Interface (preferred)

The GUI application (`ollama_side6.py`) provides a user-friendly interface for managing models:

```bash
python ollama_side6.py
```

### GUI Features

- **Model List**: Browse all installed models with their details
- **Export Tab**: Export models to a specified location
- **Import Tab**: Import models from archives
- **Delete Option**: Remove models with confirmation
- **Progress Indication**: Visual feedback during operations


## Original GUI Version (Mac OS and Tkinter argue at times)

The original Tkinter-based application (`ollama_gui.py`) is also included:

```bash
python ollama_gui.py
```

## Common Use Cases

### Backing Up Models

```bash
# Export all your models (run this for each model)
python ollama_cli.py list  # First, see what's available
python ollama_cli.py export mistral 7b ./backups/mistral-7b.tar.gz
```

### Transferring Models Between Machines

```bash
# On source machine
python ollama_cli.py export mistral 7b ./mistral-7b.tar.gz

# Transfer the file to the target machine, then:
python ollama_cli.py import ./mistral-7b.tar.gz
```

### Freeing Up Space

```bash
# List models to find large ones
python ollama_cli.py list

# Delete models you no longer need
python ollama_cli.py delete large-model 65b
```

## Technical Details

The tools interact with Ollama's model storage structure:
- Models are stored in `~/.ollama/models/`
- Manifests are in `~/.ollama/models/manifests/registry.ollama.ai/library/<model>/<param>`
- Blob files referenced in manifests are in `~/.ollama/models/blobs/`

## Troubleshooting

### Common Issues

1. **"No models found"**: Make sure Ollama is installed and you've downloaded at least one model with `ollama pull`.

2. **Import failures**: Ensure the archive is a valid model export. The archive should contain a manifest file and all referenced blobs.

3. **Permission errors**: Ensure you have write access to the Ollama directories.

### Debugging

Run the commands with the `--verbose` or `-v` flag for more detailed output:

```bash
python ollama_cli.py -v list
```

## License

[Apache 2.0 License](LICENSE)

## Acknowledgments

- [Ollama](https://ollama.com) for the amazing tool that makes local LLMs accessible
- The PySide6/Qt team for the GUI framework