#!/usr/bin/env python3
import os
import sys
import json
import shutil
import tarfile
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
import threading
import queue
import time


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


class OllamaGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        
        self.title("Ollama Model Manager")
        self.geometry("900x600")
        self.minsize(800, 500)
        
        self.manager = OllamaModelManager()
        self.model_data = {}  # Will hold the model data
        
        # Set up message queue for thread communication
        self.queue = queue.Queue()
        
        # Show initial directory information
        print("Ollama directory paths:")
        print(f"Ollama directory: {self.manager.ollama_dir}")
        print(f"Models directory: {self.manager.models_dir}")
        print(f"Manifests directory: {self.manager.manifests_dir}")
        print(f"Blobs directory: {self.manager.blobs_dir}")
        
        # Check existence of key directories
        library_path = self.manager.manifests_dir / "registry.ollama.ai" / "library"
        print(f"Library path: {library_path} (exists: {library_path.exists()})")
        
        if library_path.exists():
            print("Models found in library directory:")
            for model_dir in library_path.glob("*"):
                if model_dir.is_dir():
                    print(f"  - {model_dir.name}")
                    param_dirs = list(model_dir.glob("*"))
                    for param_dir in param_dirs:
                        print(f"    - {param_dir.name} (is_dir: {param_dir.is_dir()}, is_file: {param_dir.is_file()})")
        
        self.create_widgets()
        self.refresh_models()
        
        # Set up periodic queue check
        self.after(100, self.check_queue)
    
    def check_queue(self):
        """Check for messages from worker threads"""
        try:
            while True:
                task, args = self.queue.get_nowait()
                
                if task == "update_status":
                    message, progress = args
                    self.update_status(message, progress)
                elif task == "show_error":
                    messagebox.showerror("Error", args[0])
                elif task == "show_info":
                    messagebox.showinfo("Information", args[0])
                elif task == "refresh_models":
                    self.refresh_models()
                
                self.queue.task_done()
        except queue.Empty:
            pass
        
        # Schedule the next check
        self.after(100, self.check_queue)
    
    def create_widgets(self):
        """Create all the GUI widgets"""
        # Main frame
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Top section - Model list
        model_frame = ttk.LabelFrame(main_frame, text="Installed Models")
        model_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Button frame for the model list
        button_frame = ttk.Frame(model_frame)
        button_frame.pack(fill=tk.X, pady=5)
        
        self.refresh_btn = ttk.Button(button_frame, text="Refresh", command=self.refresh_models)
        self.refresh_btn.pack(side=tk.LEFT, padx=5)
        
        # Add Delete Model button - initially disabled until a model is selected
        self.delete_btn = ttk.Button(button_frame, text="Delete Model", command=self.delete_model, state=tk.DISABLED)
        self.delete_btn.pack(side=tk.LEFT, padx=5)
        
        # Model tree view
        columns = ("Model", "Parameter Size", "Size (GB)", "Path")
        self.model_tree = ttk.Treeview(model_frame, columns=columns, show="headings", selectmode='browse')
        
        for col in columns:
            self.model_tree.heading(col, text=col)
            if col == "Path":
                self.model_tree.column(col, width=300, stretch=True)
            elif col == "Size (GB)":
                self.model_tree.column(col, width=80, anchor=tk.E)
            else:
                self.model_tree.column(col, width=150)
        
        # Scrollbars for treeview
        vsb = ttk.Scrollbar(model_frame, orient="vertical", command=self.model_tree.yview)
        hsb = ttk.Scrollbar(model_frame, orient="horizontal", command=self.model_tree.xview)
        self.model_tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        
        # Pack the treeview and scrollbars
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        hsb.pack(side=tk.BOTTOM, fill=tk.X)
        self.model_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Bottom section - Action tabs
        action_frame = ttk.LabelFrame(main_frame, text="Actions")
        action_frame.pack(fill=tk.BOTH, padx=5, pady=5)
        
        # Tabs for different actions
        self.tabs = ttk.Notebook(action_frame)
        self.tabs.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Export tab
        export_frame = ttk.Frame(self.tabs)
        self.tabs.add(export_frame, text="Export Model")
        
        # Export widgets
        ttk.Label(export_frame, text="Select a model from the list above or enter model details:").grid(row=0, column=0, columnspan=3, sticky=tk.W, padx=5, pady=5)
        
        ttk.Label(export_frame, text="Model Name:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.export_model_var = tk.StringVar()
        ttk.Entry(export_frame, textvariable=self.export_model_var, width=20).grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        
        ttk.Label(export_frame, text="Parameter Size:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.export_param_var = tk.StringVar()
        ttk.Entry(export_frame, textvariable=self.export_param_var, width=10).grid(row=2, column=1, sticky=tk.W, padx=5, pady=5)
        
        ttk.Label(export_frame, text="Output File:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)
        self.export_output_var = tk.StringVar()
        ttk.Entry(export_frame, textvariable=self.export_output_var, width=40).grid(row=3, column=1, sticky=tk.W+tk.E, padx=5, pady=5)
        ttk.Button(export_frame, text="Browse...", command=self.browse_export_file).grid(row=3, column=2, padx=5, pady=5)
        
        ttk.Button(export_frame, text="Export Model", command=self.export_model).grid(row=4, column=0, columnspan=3, pady=10)
        
        # Import tab
        import_frame = ttk.Frame(self.tabs)
        self.tabs.add(import_frame, text="Import Model")
        
        # Import widgets
        ttk.Label(import_frame, text="Import a model from a .tar.gz file:").grid(row=0, column=0, columnspan=3, sticky=tk.W, padx=5, pady=5)
        
        ttk.Label(import_frame, text="Archive File:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.import_file_var = tk.StringVar()
        ttk.Entry(import_frame, textvariable=self.import_file_var, width=40).grid(row=1, column=1, sticky=tk.W+tk.E, padx=5, pady=5)
        ttk.Button(import_frame, text="Browse...", command=self.browse_import_file).grid(row=1, column=2, padx=5, pady=5)
        
        ttk.Label(import_frame, text="Custom Name (optional):").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.import_name_var = tk.StringVar()
        ttk.Entry(import_frame, textvariable=self.import_name_var, width=20).grid(row=2, column=1, sticky=tk.W, padx=5, pady=5)
        ttk.Label(import_frame, text="Format: name:param_size").grid(row=2, column=2, sticky=tk.W, padx=5, pady=5)
        
        ttk.Button(import_frame, text="Import Model", command=self.import_model).grid(row=3, column=0, columnspan=3, pady=10)
        
        # Status bar and progress
        status_frame = ttk.Frame(main_frame)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM, pady=5)
        
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        ttk.Label(status_frame, textvariable=self.status_var).pack(side=tk.LEFT, padx=5)
        
        # Add a spinner label for visual feedback on operations
        self.spinner_index = 0
        self.spinner_symbols = ["⣾", "⣽", "⣻", "⢿", "⡿", "⣟", "⣯", "⣷"]
        self.spinner_var = tk.StringVar()
        self.spinner_var.set("")
        self.spinner_label = ttk.Label(status_frame, textvariable=self.spinner_var, font=("Arial", 12))
        self.spinner_label.pack(side=tk.LEFT, padx=5)
        
        # Initialize operation tracking variables
        self.active_operation = False
        self.cancel_requested = False
        
        # Cancel button for long-running operations (initially hidden)
        self.cancel_button = ttk.Button(status_frame, text="Cancel", command=self.cancel_operation)
        # Will be packed when needed
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress = ttk.Progressbar(status_frame, variable=self.progress_var, length=200, mode="determinate")
        self.progress.pack(side=tk.RIGHT, padx=5)
        
        # Set up tree view selection handler
        self.model_tree.bind("<<TreeviewSelect>>", self.on_tree_select)
    
    def refresh_models(self):
        """Refresh the model list"""
        # Update status without triggering the operation UI state
        self.status_var.set("Refreshing model list...")
        self.progress_var.set(0)
        
        # Clear existing items
        for item in self.model_tree.get_children():
            self.model_tree.delete(item)
        
        # Get model data
        print("\n--- Refreshing Model List ---")
        self.model_data = self.manager.list_models()
        print(f"Found {len(self.model_data)} models")
        
        if not self.model_data:
            # Try manual detection
            print("\nAttempting manual model detection...")
            self.try_manual_detection()
        
        # Populate treeview
        for model_name, variants in self.model_data.items():
            print(f"Adding model {model_name} with {len(variants)} variants to tree view")
            for variant in variants:
                values = (
                    model_name,
                    variant["parameter_size"],
                    f"{variant['size_gb']:.2f}",
                    variant["path"]
                )
                self.model_tree.insert("", tk.END, values=values, tags=(model_name, variant["parameter_size"]))
        
        # Update status without triggering operation UI state
        self.status_var.set("Model list refreshed")
        self.progress_var.set(100)
        
    def try_manual_detection(self):
        """Try to manually detect models"""
        try:
            # Try direct path to common places
            library_path = self.manager.manifests_dir / "registry.ollama.ai" / "library"
            
            if not library_path.exists():
                print(f"Library path {library_path} does not exist, cannot manually detect models")
                return
                
            # Look for typical model directories
            print(f"Scanning library path: {library_path}")
            for model_dir in library_path.glob("*"):
                if not model_dir.is_dir():
                    continue
                    
                model_name = model_dir.name
                print(f"Found model directory: {model_name}")
                
                # Check if there are parameter size directories
                param_dirs_found = False
                for param_dir in model_dir.glob("*"):
                    param_size = param_dir.name
                    param_dirs_found = True
                    
                    print(f"  Found parameter directory/file: {param_dir}")
                    
                    # Try to add this to model data
                    if model_name not in self.model_data:
                        self.model_data[model_name] = []
                    
                    # Create a basic entry with estimated information
                    self.model_data[model_name].append({
                        "parameter_size": param_size,
                        "size_gb": 0.0,  # Unknown size
                        "path": str(param_dir),
                        "manifest_file": str(param_dir) if param_dir.is_file() else 
                                        str(param_dir / "manifest") if (param_dir / "manifest").exists() else
                                        str(param_dir / param_size) if (param_dir / param_size).exists() else
                                        str(param_dir)
                    })
                    print(f"  Added manual entry for {model_name}:{param_size}")
                
                if not param_dirs_found:
                    print(f"  No parameter directories found for {model_name}")
        except Exception as e:
            print(f"Error in manual detection: {e}")
            import traceback
            traceback.print_exc()
    
    def on_tree_select(self, event):
        """Handle tree view selection"""
        selected = self.model_tree.selection()
        if not selected:
            # Disable delete button if no selection
            self.delete_btn.config(state=tk.DISABLED)
            return
            
        # Get values from selected row
        item = self.model_tree.item(selected[0])
        values = item["values"]
        
        # Fill export fields with selected model
        if len(values) >= 2:
            self.export_model_var.set(values[0])
            self.export_param_var.set(values[1])
            self.export_output_var.set(f"{values[0]}-{values[1]}.tar.gz")
            
            # Enable delete button when a model is selected
            self.delete_btn.config(state=tk.NORMAL)
    
    def browse_export_file(self):
        """Browse for export file location"""
        try:
            model = self.export_model_var.get()
            param = self.export_param_var.get()
            default_name = f"{model}-{param}.tar.gz" if model and param else "model.tar.gz"
            
            # Use the simplest form of the dialog to avoid issues with filetypes
            filename = filedialog.asksaveasfilename(
                title="Save Model Archive"
            )
            
            # Only update if a valid filename was selected
            if filename and isinstance(filename, str):
                # Add .tar.gz extension if not present
                if not filename.endswith('.tar.gz'):
                    filename += '.tar.gz'
                self.export_output_var.set(filename)
        except Exception as e:
            print(f"Error in file dialog: {e}")
            messagebox.showerror("Error", f"Could not open file dialog: {e}")
    
    def browse_import_file(self):
        """Browse for import file"""
        try:
            # Use the simplest form of the dialog to avoid issues with filetypes
            filename = filedialog.askopenfilename(
                title="Select Model Archive"
            )
            
            # Only update if a valid filename was selected
            if filename and isinstance(filename, str):
                self.import_file_var.set(filename)
                
                # Try to extract model name from filename
                basename = os.path.basename(filename)
                if basename.endswith(".tar.gz"):
                    basename = basename[:-7]  # Remove .tar.gz
                    
                    # Try to parse model-param pattern
                    parts = basename.split("-")
                    if len(parts) >= 2:
                        model_name = "-".join(parts[:-1])  # Everything except last part
                        param_size = parts[-1]  # Last part
                        
                        self.import_name_var.set(f"{model_name}:{param_size}")
                    else:
                        # Handle simple case
                        self.import_name_var.set(f"{basename}:default")
        except Exception as e:
            print(f"Error in file dialog: {e}")
            messagebox.showerror("Error", f"Could not open file dialog: {e}")
    
    def export_model(self):
        """Export the selected model"""
        model_name = self.export_model_var.get().strip()
        param_size = self.export_param_var.get().strip()
        output_path = self.export_output_var.get().strip()
        
        if not model_name or not param_size:
            messagebox.showerror("Error", "Please specify both model name and parameter size")
            return
            
        if not output_path:
            messagebox.showerror("Error", "Please specify an output file path")
            return
        
        # Find the manifest file
        manifest_file = None
        
        # Check if we have this model in our data
        if model_name in self.model_data:
            for variant in self.model_data[model_name]:
                if variant["parameter_size"] == param_size:
                    manifest_file = variant["manifest_file"]
                    break
        
        # If not found in model data, try direct path
        if not manifest_file:
            # Try direct path to model (param_size is the file itself)
            library_path = self.manager.manifests_dir / "registry.ollama.ai" / "library"
            model_path = library_path / model_name / param_size
            
            # In the correct structure, model_path is the manifest file itself
            if model_path.exists() and model_path.is_file():
                manifest_file = model_path
            # For backward compatibility - try common alternatives if it's a directory
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
            messagebox.showerror("Error", f"Could not find manifest for model {model_name}:{param_size}")
            return
            
        # Start export in a separate thread
        self.update_status(f"Exporting {model_name}:{param_size}...", 0)
        thread = threading.Thread(
            target=self._export_thread, 
            args=(manifest_file, model_name, param_size, output_path)
        )
        thread.daemon = True
        thread.start()
    
    def _export_thread(self, manifest_file, model_name, param_size, output_path):
        """Thread function for model export"""
        # Create a wrapper for progress callback that checks for cancellation
        def progress_callback(msg, pct):
            # Check if the operation was cancelled
            if self.cancel_requested:
                return False  # Signal to stop the operation
            # Otherwise, update progress normally
            self.queue.put(("update_status", (msg, pct)))
            return True  # Continue operation
        
        try:
            # Enable UI indication of operation in progress
            self.queue.put(("update_status", (f"Starting export of {model_name}:{param_size}...", 0)))
            
            error = self.manager.export_model(
                manifest_file, 
                model_name, 
                param_size, 
                output_path,
                progress_callback
            )
            
            # Check again for cancellation
            if self.cancel_requested:
                self.queue.put(("show_info", ("Export operation was cancelled",)))
                self.queue.put(("update_status", ("Export cancelled", 100)))
                return
                
            if error:
                self.queue.put(("show_error", (error,)))
                self.queue.put(("update_status", ("Export failed", 100)))
            else:
                self.queue.put(("show_info", (f"Model exported successfully to {output_path}",)))
                self.queue.put(("update_status", ("Export complete", 100)))
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.queue.put(("show_error", (f"Error during export: {e}",)))
            self.queue.put(("update_status", ("Export failed", 100)))
    
    def import_model(self):
        """Import a model from an archive"""
        archive_path = self.import_file_var.get().strip()
        custom_name = self.import_name_var.get().strip()
        
        if not archive_path:
            messagebox.showerror("Error", "Please select an archive file")
            return
            
        if not os.path.exists(archive_path):
            messagebox.showerror("Error", f"File not found: {archive_path}")
            return
        
        # Start import in a separate thread
        self.update_status("Importing model...", 0)
        thread = threading.Thread(
            target=self._import_thread, 
            args=(archive_path, custom_name)
        )
        thread.daemon = True
        thread.start()
    
    def _import_thread(self, archive_path, custom_name):
        """Thread function for model import"""
        # Create a wrapper for progress callback that checks for cancellation
        def progress_callback(msg, pct):
            # Check if the operation was cancelled
            if self.cancel_requested:
                return False  # Signal to stop the operation
            # Otherwise, update progress normally
            self.queue.put(("update_status", (msg, pct)))
            return True  # Continue operation
        
        try:
            # Enable UI indication of operation in progress
            self.queue.put(("update_status", ("Starting model import...", 0)))
            
            error = self.manager.import_model(
                archive_path, 
                custom_name,
                progress_callback
            )
            
            # Check again for cancellation
            if self.cancel_requested:
                self.queue.put(("show_info", ("Import operation was cancelled",)))
                self.queue.put(("update_status", ("Import cancelled", 100)))
                return
                
            if error:
                self.queue.put(("show_error", (error,)))
                self.queue.put(("update_status", ("Import failed", 100)))
            else:
                self.queue.put(("show_info", ("Model imported successfully",)))
                self.queue.put(("update_status", ("Import complete", 100)))
                self.queue.put(("refresh_models", ()))
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.queue.put(("show_error", (f"Error during import: {e}",)))
            self.queue.put(("update_status", ("Import failed", 100)))
    
    def delete_model(self):
        """Delete the selected model"""
        selected = self.model_tree.selection()
        if not selected:
            return
            
        # Get values from selected row
        item = self.model_tree.item(selected[0])
        values = item["values"]
        
        if len(values) < 2:
            return
            
        model_name = values[0]
        param_size = values[1]
        
        # Confirm deletion
        confirm = messagebox.askyesno(
            "Confirm Delete", 
            f"Are you sure you want to delete model {model_name}:{param_size}?\n\n"
            "This will delete the model manifest and associated blob files.",
            icon=messagebox.WARNING
        )
        
        if not confirm:
            return
            
        # First, clear the selection to avoid potential issues with deleted items
        self.model_tree.selection_remove(selected)
        self.delete_btn.config(state=tk.DISABLED)
        
        # Clear any export/import fields that might reference the model
        if self.export_model_var.get() == model_name and self.export_param_var.get() == param_size:
            self.export_model_var.set("")
            self.export_param_var.set("")
            self.export_output_var.set("")
            
        # Start deletion in a separate thread
        self.update_status(f"Deleting {model_name}:{param_size}...", 0)
        thread = threading.Thread(
            target=self._delete_thread, 
            args=(model_name, param_size)
        )
        thread.daemon = True
        thread.start()
    
    def _delete_thread(self, model_name, param_size):
        """Thread function for model deletion"""
        # Create a wrapper for progress callback that checks for cancellation
        def progress_callback(msg, pct):
            # Check if the operation was cancelled
            if self.cancel_requested:
                return False  # Signal to stop the operation
            # Otherwise, update progress normally
            self.queue.put(("update_status", (msg, pct)))
            return True  # Continue operation
            
        try:
            # Enable UI indication of operation in progress
            self.queue.put(("update_status", (f"Starting deletion of {model_name}:{param_size}...", 0)))
            
            error = self.manager.delete_model(
                model_name,
                param_size,
                progress_callback
            )
            
            # Check for cancellation
            if self.cancel_requested:
                self.queue.put(("show_info", ("Delete operation was cancelled",)))
                self.queue.put(("update_status", ("Delete cancelled", 100)))
                return
            
            if error:
                self.queue.put(("show_error", (error,)))
                self.queue.put(("update_status", ("Delete failed", 100)))
            else:
                # Remove the model from our internal data structure immediately
                if model_name in self.model_data:
                    # Filter out the deleted variant
                    self.model_data[model_name] = [
                        variant for variant in self.model_data[model_name] 
                        if variant["parameter_size"] != param_size
                    ]
                    
                    # If no variants left, remove the model entry
                    if not self.model_data[model_name]:
                        del self.model_data[model_name]
                        
                self.queue.put(("show_info", (f"Model {model_name}:{param_size} deleted successfully",)))
                self.queue.put(("update_status", ("Delete complete", 100)))
                # Queue a complete refresh to ensure UI is in sync
                self.queue.put(("refresh_models", ()))
        except Exception as e:
            # Catch any unexpected errors during deletion
            import traceback
            traceback.print_exc()
            self.queue.put(("show_error", (f"Error during deletion: {e}",)))
            self.queue.put(("update_status", ("Delete failed", 100)))
    
    def update_status(self, message, progress):
        """Update status bar and progress"""
        self.status_var.set(message)
        self.progress_var.set(progress)
        
        # Handle spinner animation based on progress
        if progress < 100:
            # Operation in progress, show spinner and ensure cancel button is visible
            if not self.active_operation:
                self.start_operation()
            # Update spinner animation
            self.update_spinner()
        else:
            # Operation complete, hide spinner and cancel button
            self.end_operation()
    
    def update_spinner(self):
        """Update the spinner animation"""
        self.spinner_index = (self.spinner_index + 1) % len(self.spinner_symbols)
        self.spinner_var.set(self.spinner_symbols[self.spinner_index])
        # Schedule the next spinner update if operation is still active
        if self.active_operation:
            self.after(100, self.update_spinner)
    
    def start_operation(self):
        """Start a long-running operation"""
        self.active_operation = True
        self.cancel_requested = False
        # Show spinner
        self.spinner_var.set(self.spinner_symbols[0])
        # Show cancel button
        self.cancel_button.pack(side=tk.RIGHT, padx=5)
        # Start spinner animation
        self.update_spinner()
        # Disable UI buttons and tab-switching (instead of disabling the tabs themselves)
        for child in self.tabs.winfo_children():
            for widget in child.winfo_children():
                if isinstance(widget, ttk.Button):
                    widget.config(state=tk.DISABLED)
        # Disable model tree interaction
        self.model_tree.config(selectmode='none')
        # Disable the refresh and delete buttons
        self.refresh_btn.config(state=tk.DISABLED)
        if hasattr(self, 'delete_btn'):
            self.delete_btn.config(state=tk.DISABLED)
        
    def end_operation(self):
        """End a long-running operation"""
        self.active_operation = False
        # Hide spinner
        self.spinner_var.set("")
        # Hide cancel button
        self.cancel_button.pack_forget()
        # Re-enable UI elements
        for child in self.tabs.winfo_children():
            for widget in child.winfo_children():
                if isinstance(widget, ttk.Button):
                    widget.config(state=tk.NORMAL)
        # Re-enable model tree interaction
        self.model_tree.config(selectmode='browse')
        # Re-enable the refresh button
        self.refresh_btn.config(state=tk.NORMAL)
        # The delete button state depends on selection, so we don't enable it here
        
    def cancel_operation(self):
        """Cancel the current operation"""
        if self.active_operation:
            self.cancel_requested = True
            self.update_status("Cancelling operation...", 0)
            # The thread will need to check cancel_requested periodically


if __name__ == "__main__":
    app = OllamaGUI()
    app.mainloop()