#!/usr/bin/env python3
import os
import sys
import json
import shutil
import tarfile
from pathlib import Path
import argparse
import threading
import time
from typing import Optional, List, Dict, Any, Callable


class OllamaModelManager:
    def __init__(self):
        self.ollama_dir = Path.home() / ".ollama"
        
        # Check if we're using the newer structure with models subdirectory
        self.models_dir = self.ollama_dir / "models"
        if self.models_dir.exists():
            self.manifests_dir = self.models_dir / "manifests"
            self.blobs_dir = self.models_dir / "blobs"
        else:
            # Fall back to older structure where manifests and blobs are directly under .ollama
            self.models_dir = self.ollama_dir  # For backward compatibility
            self.manifests_dir = self.ollama_dir / "manifests"
            self.blobs_dir = self.ollama_dir / "blobs"
    
    def list_models(self):
        """List all models available in the Ollama directory."""
        models = {}
        
        # Get the library path
        library_path = self.manifests_dir / "registry.ollama.ai" / "library"
        print(f"Looking for models in: {library_path}")
        if not library_path.exists():
            print(f"Library path does not exist: {library_path}")
            return models
            
        # Walk through model directories
        for model_dir in library_path.glob("*"):
            print(f"Examining potential model directory: {model_dir}")
            if not model_dir.is_dir():
                print(f"  Not a directory, skipping")
                continue
                
            model_name = model_dir.name
            print(f"  Found model directory: {model_name}")
            
            # Check each parameter size file or directory
            for model_path in model_dir.glob("*"):
                print(f"    Examining parameter path: {model_path}")
                print(f"    Is directory: {model_path.is_dir()}, Is file: {model_path.is_file()}")
                
                param_size = model_path.name
                manifest_file = None
                
                # The proper structure: param_size should be a file directly
                # If model_path is a file, it is the manifest itself
                if model_path.is_file():
                    manifest_file = model_path
                    print(f"    Found manifest file at: {manifest_file}")
                # For backward compatibility - if it's a directory, look for manifest files inside
                elif model_path.is_dir():
                    alternatives = [
                        model_path / param_size,
                        model_path / "manifest",
                        model_path / "manifest.json"
                    ]
                    
                    print(f"    Looking for manifest files in directory {model_path} (legacy structure)")
                    for alt in alternatives:
                        print(f"      Checking: {alt} (exists: {alt.exists()})")
                        if alt.exists() and alt.is_file():
                            manifest_file = alt
                            print(f"      Found manifest file at: {manifest_file}")
                            break
                
                if manifest_file is not None and manifest_file.exists() and manifest_file.is_file():
                    try:
                        print(f"    Reading manifest file: {manifest_file}")
                        with open(manifest_file, 'r') as f:
                            manifest_content = f.read()
                            try:
                                manifest = json.loads(manifest_content)
                                print(f"    Successfully parsed manifest JSON")
                                
                                # Get model size by summing layer sizes
                                layers = manifest.get("layers", [])
                                print(f"    Found {len(layers)} layers in manifest")
                                model_size = sum(layer.get("size", 0) for layer in layers)
                                model_size_gb = model_size / (1024 ** 3)
                                
                                # Create or update model entry
                                if model_name not in models:
                                    models[model_name] = []
                                
                                models[model_name].append({
                                    "parameter_size": param_size,
                                    "size_gb": model_size_gb,
                                    "path": str(model_path),
                                    "manifest_file": str(manifest_file)
                                })
                                print(f"    Added model {model_name}:{param_size} to list")
                            except json.JSONDecodeError as e:
                                print(f"    Error parsing JSON: {e}")
                                print(f"    First 100 characters of file: {manifest_content[:100]}")
                    except Exception as e:
                        print(f"Error reading manifest for {model_path}: {e}", file=sys.stderr)
        
        return models
    
    def export_model(self, manifest_file, model_name, param_size, output_path, progress_callback=None):
        """Export a model to a tar.gz file."""
        manifest_file = Path(manifest_file)
        
        # Read the manifest
        try:
            with open(manifest_file, 'r') as f:
                manifest = json.load(f)
        except Exception as e:
            return f"Error reading manifest: {e}"
        
        # Create temporary directory for export
        temp_dir = Path(f"/tmp/ollama-export-{model_name}-{param_size}")
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        temp_dir.mkdir(parents=True)
        
        # Create metadata file with model info
        metadata = {
            "model_name": model_name,
            "parameter_size": param_size,
            "original_path": str(manifest_file.parent)
        }
        with open(temp_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Copy manifest
        manifest_dest = temp_dir / "manifest"
        shutil.copy2(manifest_file, manifest_dest)
        
        # Get total size for progress calculation
        total_size = 0
        for layer in manifest.get("layers", []):
            total_size += layer.get("size", 0)
        
        # Add config size
        config_size = manifest.get("config", {}).get("size", 0)
        total_size += config_size
        
        # Copy all blob files
        blob_files = []
        current_size = 0
        
        # Process each layer in the manifest
        for i, layer in enumerate(manifest.get("layers", [])):
            digest = layer.get("digest", "")
            if digest.startswith("sha256:"):
                # Convert from "sha256:abc123" to "sha256-abc123"
                blob_filename = digest.replace(":", "-")
                blob_path = self.blobs_dir / blob_filename
                
                if blob_path.exists():
                    # Copy blob to temp dir
                    size_mb = layer.get("size", 0) / (1024 ** 2)
                    if progress_callback:
                        if not progress_callback(f"Copying {blob_filename} ({size_mb:.2f} MB)", (current_size / total_size) * 100):
                            # Operation was cancelled
                            shutil.rmtree(temp_dir, ignore_errors=True)
                            return "Operation cancelled by user"
                    
                    shutil.copy2(blob_path, temp_dir / blob_filename)
                    blob_files.append(blob_filename)
                    
                    # Update progress
                    current_size += layer.get("size", 0)
                    if progress_callback:
                        if not progress_callback(f"Copied {blob_filename}", (current_size / total_size) * 100):
                            # Operation was cancelled
                            shutil.rmtree(temp_dir, ignore_errors=True)
                            return "Operation cancelled by user"
                else:
                    return f"Error: Blob file {blob_filename} not found"
        
        # Also copy the config blob
        config_digest = manifest.get("config", {}).get("digest", "")
        if config_digest.startswith("sha256:"):
            blob_filename = config_digest.replace(":", "-")
            blob_path = self.blobs_dir / blob_filename
            
            if blob_path.exists():
                if progress_callback:
                    if not progress_callback(f"Copying config {blob_filename}", (current_size / total_size) * 100):
                        # Operation was cancelled
                        shutil.rmtree(temp_dir, ignore_errors=True)
                        return "Operation cancelled by user"
                
                shutil.copy2(blob_path, temp_dir / blob_filename)
                blob_files.append(blob_filename)
                
                # Update progress
                current_size += config_size
                if progress_callback:
                    if not progress_callback(f"Copied config {blob_filename}", (current_size / total_size) * 100):
                        # Operation was cancelled
                        shutil.rmtree(temp_dir, ignore_errors=True)
                        return "Operation cancelled by user"
        
        # Create the tar.gz file
        if progress_callback:
            if not progress_callback("Creating archive...", 95):
                # Operation was cancelled
                shutil.rmtree(temp_dir, ignore_errors=True)
                return "Operation cancelled by user"
            
        try:
            with tarfile.open(output_path, "w:gz") as tar:
                # Add metadata file
                tar.add(temp_dir / "metadata.json", arcname="metadata.json")
                
                # Add manifest file
                tar.add(temp_dir / "manifest", arcname="manifest")
                
                # Add all blob files
                for blob_file in blob_files:
                    # Check for cancellation during archiving
                    if progress_callback and not progress_callback(f"Adding {blob_file} to archive...", 95):
                        shutil.rmtree(temp_dir, ignore_errors=True)
                        # If the archive file was partially created, delete it
                        if os.path.exists(output_path):
                            os.unlink(output_path)
                        return "Operation cancelled by user"
                    tar.add(temp_dir / blob_file, arcname=blob_file)
        except Exception as e:
            # Clean up any partial files on error
            shutil.rmtree(temp_dir, ignore_errors=True)
            if os.path.exists(output_path):
                os.unlink(output_path)
            return f"Error creating archive: {e}"
        
        # Clean up
        shutil.rmtree(temp_dir, ignore_errors=True)
        
        if progress_callback:
            progress_callback("Export complete", 100)
        
        return None  # No error
    
    def import_model(self, archive_path, custom_name=None, progress_callback=None):
        """Import a model from a tar.gz file."""
        archive_path = Path(archive_path)
        
        if not archive_path.exists():
            return f"Error: Archive {archive_path} not found"
        
        # Create temporary directory for import
        temp_dir = Path(f"/tmp/ollama-import-{archive_path.stem}")
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        temp_dir.mkdir(parents=True)
        
        # Extract the archive
        if progress_callback:
            if not progress_callback("Extracting archive...", 10):
                # Operation was cancelled
                shutil.rmtree(temp_dir)
                return "Operation cancelled by user"
            
        try:
            with tarfile.open(archive_path, "r:gz") as tar:
                # Extract files with cancellation check
                for member in tar.getmembers():
                    if progress_callback and not progress_callback(f"Extracting {member.name}...", 10):
                        # Operation was cancelled
                        shutil.rmtree(temp_dir)
                        return "Operation cancelled by user"
                    tar.extract(member, path=temp_dir)
        except Exception as e:
            shutil.rmtree(temp_dir)
            return f"Error extracting archive: {e}"
        
        # Check for manifest file
        manifest_file = temp_dir / "manifest"
        metadata_file = temp_dir / "metadata.json"
        
        if not manifest_file.exists():
            shutil.rmtree(temp_dir)
            return "Error: No manifest file found in archive"
        
        # Read the manifest
        if progress_callback:
            if not progress_callback("Reading manifest...", 15):
                # Operation was cancelled
                shutil.rmtree(temp_dir)
                return "Operation cancelled by user"
                
        try:
            with open(manifest_file, 'r') as f:
                manifest = json.load(f)
        except Exception as e:
            shutil.rmtree(temp_dir)
            return f"Error reading manifest: {e}"
        
        # Determine model name and parameter size
        if metadata_file.exists():
            # Use metadata if available
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                model_name = metadata.get("model_name")
                param_size = metadata.get("parameter_size")
            except Exception as e:
                return f"Error reading metadata: {e}"
        else:
            model_name = None
            param_size = None
        
        # Use custom name if provided
        if custom_name:
            parts = custom_name.split(':')
            model_name = parts[0]
            param_size = parts[1] if len(parts) > 1 else "default"
        # If still no name, infer from archive name
        elif not model_name or not param_size:
            # Try to infer from filename: name-param.tar.gz
            parts = archive_path.stem.split('-')
            if len(parts) > 1:
                model_name = parts[0]
                param_size = parts[-1]
            else:
                model_name = archive_path.stem
                param_size = "default"
        
        # Create parent directory (model_name), but not parameter size (which is a file)
        model_dir = self.manifests_dir / "registry.ollama.ai" / "library" / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # The model_path should be a file with parameter size as the filename
        model_path = model_dir / param_size
        
        if progress_callback:
            if not progress_callback(f"Importing {model_name}:{param_size}...", 20):
                # Operation was cancelled
                shutil.rmtree(temp_dir)
                return "Operation cancelled by user"
        
        # Copy manifest to model path with the parameter size as the filename
        try:
            shutil.copy2(manifest_file, model_path)
        except Exception as e:
            shutil.rmtree(temp_dir)
            return f"Error copying manifest: {e}"
        
        # Get total size for progress calculation
        total_size = sum(layer.get("size", 0) for layer in manifest.get("layers", []))
        total_size += manifest.get("config", {}).get("size", 0)
        
        # Copy all blob files
        current_size = 0
        
        # Copy config blob
        config_digest = manifest.get("config", {}).get("digest", "")
        if config_digest.startswith("sha256:"):
            blob_filename = config_digest.replace(":", "-")
            source_path = temp_dir / blob_filename
            dest_path = self.blobs_dir / blob_filename
            
            if source_path.exists():
                if progress_callback:
                    if not progress_callback(f"Copying config {blob_filename}", 25):
                        # Operation was cancelled
                        shutil.rmtree(temp_dir)
                        return "Operation cancelled by user"
                
                try:    
                    shutil.copy2(source_path, dest_path)
                    current_size += manifest.get("config", {}).get("size", 0)
                except Exception as e:
                    shutil.rmtree(temp_dir)
                    return f"Error copying config blob: {e}"
        
        # Copy layer blobs
        for i, layer in enumerate(manifest.get("layers", [])):
            digest = layer.get("digest", "")
            if digest.startswith("sha256:"):
                blob_filename = digest.replace(":", "-")
                source_path = temp_dir / blob_filename
                dest_path = self.blobs_dir / blob_filename
                
                if source_path.exists():
                    size_mb = layer.get("size", 0) / (1024 ** 2)
                    progress_pct = 25 + (i / len(manifest.get("layers", []))) * 70
                    
                    if progress_callback:
                        if not progress_callback(f"Copying {blob_filename} ({size_mb:.2f} MB)", progress_pct):
                            # Operation was cancelled
                            shutil.rmtree(temp_dir)
                            return "Operation cancelled by user"
                    
                    try:
                        shutil.copy2(source_path, dest_path)
                        current_size += layer.get("size", 0)
                    except Exception as e:
                        shutil.rmtree(temp_dir)
                        return f"Error copying layer blob: {e}"
                else:
                    shutil.rmtree(temp_dir)
                    return f"Error: Blob file {blob_filename} not found in archive"
        
        # Clean up
        shutil.rmtree(temp_dir)
        
        if progress_callback:
            if not progress_callback("Import complete", 100):
                # Even if cancelled at the very end, we've already done everything
                return "Operation completed but user cancelled at final step"
            
        return None  # No error
    
    def delete_model(self, model_name, param_size, progress_callback=None):
        """Delete a model and its associated blobs."""
        # Get the model path
        model_path = self.manifests_dir / "registry.ollama.ai" / "library" / model_name / param_size
        
        # Check if the model exists
        if not model_path.exists():
            return f"Error: Model {model_name}:{param_size} not found"
        
        # Read the manifest to find associated blob files
        if progress_callback:
            if not progress_callback(f"Reading manifest for {model_name}:{param_size}...", 10):
                return "Operation cancelled by user"
        
        try:
            with open(model_path, 'r') as f:
                manifest = json.load(f)
        except Exception as e:
            return f"Error reading manifest: {e}"
        
        # Collect all blob digests
        blob_digests = []
        
        # Add config digest
        config_digest = manifest.get("config", {}).get("digest", "")
        if config_digest.startswith("sha256:"):
            blob_digests.append(config_digest)
        
        # Add layer digests
        for layer in manifest.get("layers", []):
            digest = layer.get("digest", "")
            if digest.startswith("sha256:"):
                blob_digests.append(digest)
        
        # Convert digests to filenames
        blob_filenames = [digest.replace(":", "-") for digest in blob_digests]
        
        # Delete blob files
        if progress_callback:
            if not progress_callback("Deleting blob files...", 30):
                return "Operation cancelled by user"
        
        for i, blob_filename in enumerate(blob_filenames):
            blob_path = self.blobs_dir / blob_filename
            
            if blob_path.exists():
                progress_pct = 30 + (i / len(blob_filenames)) * 60
                
                if progress_callback:
                    if not progress_callback(f"Deleting {blob_filename}", progress_pct):
                        return "Operation cancelled by user"
                
                try:
                    blob_path.unlink()
                except Exception as e:
                    print(f"Warning: Could not delete blob {blob_filename}: {e}", file=sys.stderr)
        
        # Delete the manifest file
        if progress_callback:
            if not progress_callback("Deleting manifest...", 90):
                return "Operation cancelled by user"
        
        try:
            model_path.unlink()
        except Exception as e:
            return f"Error deleting manifest: {e}"
        
        # Check if model directory is empty and delete if it is
        model_dir = self.manifests_dir / "registry.ollama.ai" / "library" / model_name
        try:
            remaining_files = list(model_dir.glob("*"))
            if not remaining_files:
                if progress_callback:
                    if not progress_callback(f"Deleting empty model directory {model_name}...", 95):
                        # We've already deleted the manifest, so proceed even if cancelled
                        pass
                model_dir.rmdir()
        except Exception as e:
            print(f"Warning: Could not check/delete model directory: {e}", file=sys.stderr)
        
        if progress_callback:
            if not progress_callback("Delete complete", 100):
                # We've completed the operation anyway
                return "Operation completed but user cancelled at final step"
            
        return None  # No error


class OllamaCLI:
    def __init__(self):
        self.manager = OllamaModelManager()
        self.model_data = {}  # Will hold the model data
        
        # Show initial directory information (only in verbose mode)
        if self.is_verbose():
            print("Ollama directory paths:")
            print(f"Ollama directory: {self.manager.ollama_dir}")
            print(f"Models directory: {self.manager.models_dir}")
            print(f"Manifests directory: {self.manager.manifests_dir}")
            print(f"Blobs directory: {self.manager.blobs_dir}")
            
            library_path = self.manager.manifests_dir / "registry.ollama.ai" / "library"
            print(f"Library path: {library_path} (exists: {library_path.exists()})")
    
    def is_verbose(self) -> bool:
        """Check if verbose mode is enabled"""
        return '--verbose' in sys.argv or '-v' in sys.argv
    
    def print_progress(self, message: str, progress: int = None):
        """Print progress message with optional percentage"""
        if progress is not None:
            print(f"{message} ({progress}%)", end="\r", flush=True)
        else:
            print(f"{message}", end="\r", flush=True)
    
    def progress_callback(self, message: str, progress: int):
        """Progress callback for long operations"""
        self.print_progress(message, progress)
        return True  # Continue operation
    
    def list_models(self, format_json=False):
        """List all available models"""
        print("Fetching models...")
        self.model_data = self.manager.list_models()
        
        if not self.model_data:
            print("No models found.")
            return
        
        if format_json:
            # Output as JSON
            print(json.dumps(self.model_data, indent=2))
        else:
            # Output as formatted table
            print("\nInstalled Models:")
            print(f"{'Model Name':<20} {'Parameter Size':<15} {'Size':<10} {'Path':<50}")
            print("-" * 95)
            
            for model_name, variants in self.model_data.items():
                for variant in variants:
                    print(f"{model_name:<20} {variant['parameter_size']:<15} {variant['size_gb']:.2f} GB    {variant['path']}")
        
        return self.model_data
        
    def export_model(self, model_name, param_size, output_path):
        """Export a model to a tar.gz file"""
        if not model_name or not param_size:
            print("Error: Please specify both model name and parameter size")
            return False
            
        if not output_path:
            print("Error: Please specify an output file path")
            return False
        
        # Find the manifest file
        manifest_file = None
        
        # Get models first if needed
        if not self.model_data:
            self.model_data = self.manager.list_models()
        
        # Check if we have this model in our data
        if model_name in self.model_data:
            for variant in self.model_data[model_name]:
                if variant["parameter_size"] == param_size:
                    manifest_file = variant["manifest_file"]
                    break
        
        # If not found in model data, try direct path
        if not manifest_file:
            # Try direct path to model
            library_path = self.manager.manifests_dir / "registry.ollama.ai" / "library"
            model_path = library_path / model_name / param_size
            
            # The manifest should be the parameter file directly
            if model_path.exists() and model_path.is_file():
                manifest_file = model_path
            # For backward compatibility - try if it's a directory
            elif model_path.exists() and model_path.is_dir():
                alternatives = [
                    model_path / param_size,
                    model_path / "manifest",
                    model_path / "manifest.json"
                ]
                
                for alt in alternatives:
                    if alt.exists() and alt.is_file():
                        manifest_file = alt
                        break
        
        if not manifest_file:
            print(f"Error: Could not find manifest for model {model_name}:{param_size}")
            return False
            
        print(f"Exporting {model_name}:{param_size} to {output_path}...")
        error = self.manager.export_model(
            manifest_file, 
            model_name, 
            param_size, 
            output_path,
            self.progress_callback
        )
        
        if error:
            print(f"\nError: {error}")
            return False
        else:
            print(f"\nModel exported successfully to {output_path}")
            return True
    
    def import_model(self, archive_path, custom_name=None):
        """Import a model from an archive"""
        if not archive_path:
            print("Error: Please specify an archive file")
            return False
            
        if not os.path.exists(archive_path):
            print(f"Error: File not found: {archive_path}")
            return False
        
        print(f"Importing model from {archive_path}...")
        if custom_name:
            print(f"Using custom name: {custom_name}")
            
        error = self.manager.import_model(
            archive_path, 
            custom_name,
            self.progress_callback
        )
        
        if error:
            print(f"\nError: {error}")
            return False
        else:
            print("\nModel imported successfully")
            return True
    
    def delete_model(self, model_name, param_size):
        """Delete a model"""
        if not model_name or not param_size:
            print("Error: Please specify both model name and parameter size")
            return False
        
        # Confirm deletion
        confirm = input(f"Are you sure you want to delete model {model_name}:{param_size}? (y/N): ")
        if confirm.lower() not in ('y', 'yes'):
            print("Deletion cancelled.")
            return False
        
        print(f"Deleting {model_name}:{param_size}...")
        error = self.manager.delete_model(
            model_name,
            param_size,
            self.progress_callback
        )
        
        if error:
            print(f"\nError: {error}")
            return False
        else:
            print(f"\nModel {model_name}:{param_size} deleted successfully")
            return True
    
def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Ollama Model Manager CLI Tool")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List available models")
    list_parser.add_argument("--json", action="store_true", help="Output as JSON")
    
    # Export command
    export_parser = subparsers.add_parser("export", help="Export a model to a tar.gz file")
    export_parser.add_argument("model", help="Model name (e.g., mistral)")
    export_parser.add_argument("parameter_size", help="Parameter size (e.g., 7b)")
    export_parser.add_argument("output_path", help="Output file path (.tar.gz)")
    export_parser.add_argument("--force", "-f", action="store_true", help="Overwrite output file if it exists")
    
    # Import command
    import_parser = subparsers.add_parser("import", help="Import a model from a tar.gz file")
    import_parser.add_argument("archive_path", help="Path to model archive (.tar.gz)")
    import_parser.add_argument("--name", help="Custom name for the imported model (format: name:param_size)")
    
    # Delete command
    delete_parser = subparsers.add_parser("delete", help="Delete a model")
    delete_parser.add_argument("model", help="Model name (e.g., mistral)")
    delete_parser.add_argument("parameter_size", help="Parameter size (e.g., 7b)")
    delete_parser.add_argument("--force", "-f", action="store_true", help="Skip confirmation prompt")
    
    # Common options
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    
    return parser.parse_args()


def main():
    """Main entry point for the CLI tool"""
    args = parse_arguments()
    cli = OllamaCLI()
    
    if args.command == "list":
        cli.list_models(format_json=args.json)
    
    elif args.command == "export":
        # Check if output file exists and force flag is not set
        output_path = args.output_path
        if os.path.exists(output_path) and not args.force:
            override = input(f"File {output_path} already exists. Overwrite? (y/N): ")
            if override.lower() not in ('y', 'yes'):
                print("Export cancelled.")
                return 1
        
        # Export the model
        success = cli.export_model(args.model, args.parameter_size, output_path)
        return 0 if success else 1
    
    elif args.command == "import":
        # Import the model
        success = cli.import_model(args.archive_path, args.name)
        return 0 if success else 1
    
    elif args.command == "delete":
        # Delete the model with optional force flag
        if args.force:
            # Skip confirmation for --force
            print(f"Forcefully deleting model {args.model}:{args.parameter_size}...")
            error = cli.manager.delete_model(
                args.model,
                args.parameter_size,
                cli.progress_callback
            )
            
            if error:
                print(f"\nError: {error}")
                return 1
            else:
                print(f"\nModel {args.model}:{args.parameter_size} deleted successfully")
                return 0
        else:
            # Normal deletion with confirmation
            success = cli.delete_model(args.model, args.parameter_size)
            return 0 if success else 1
    
    else:
        # No command specified, show help
        print("Please specify a command. Use --help for more information.")
        return 1
    
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(130)