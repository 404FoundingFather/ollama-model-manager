#!/usr/bin/env python3
import os
import sys
import json
import shutil
import tarfile
from pathlib import Path
import threading
import queue
import time

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QPushButton, QLineEdit, QTreeWidget, QTreeWidgetItem,
    QProgressBar, QFileDialog, QMessageBox, QTabWidget, QFrame, 
    QGroupBox, QGridLayout, QHeaderView, QSplitter
)
from PySide6.QtCore import Qt, QTimer, Signal, Slot
from PySide6.QtGui import QFont, QIcon


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


class OllamaGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("Ollama Model Manager")
        self.resize(900, 600)
        self.setMinimumSize(800, 500)
        
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
        
        # Create main widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        
        self.create_widgets()
        self.refresh_models()
        
        # Set up periodic queue check
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.check_queue)
        self.timer.start(100)
    
    def check_queue(self):
        """Check for messages from worker threads"""
        try:
            while True:
                task, args = self.queue.get_nowait()
                
                if task == "update_status":
                    message, progress = args
                    self.update_status(message, progress)
                elif task == "show_error":
                    QMessageBox.critical(self, "Error", args[0])
                elif task == "show_info":
                    QMessageBox.information(self, "Information", args[0])
                elif task == "refresh_models":
                    self.refresh_models()
                
                self.queue.task_done()
        except queue.Empty:
            pass
        
        # The timer will trigger the next check automatically
    
    def create_widgets(self):
        """Create all the GUI widgets"""
        # Top section - Model list
        model_group = QGroupBox("Installed Models")
        self.main_layout.addWidget(model_group)
        model_layout = QVBoxLayout(model_group)
        
        # Button frame for the model list
        button_layout = QHBoxLayout()
        model_layout.addLayout(button_layout)
        
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self.refresh_models)
        button_layout.addWidget(self.refresh_btn)
        
        # Add Delete Model button - initially disabled until a model is selected
        self.delete_btn = QPushButton("Delete Model")
        self.delete_btn.clicked.connect(self.delete_model)
        self.delete_btn.setEnabled(False)
        button_layout.addWidget(self.delete_btn)
        
        button_layout.addStretch()
        
        # Model tree view
        self.model_tree = QTreeWidget()
        self.model_tree.setHeaderLabels(["Model", "Parameter Size", "Size (GB)", "Path"])
        self.model_tree.setSelectionMode(QTreeWidget.SelectionMode.SingleSelection)
        self.model_tree.setAlternatingRowColors(True)
        self.model_tree.itemSelectionChanged.connect(self.on_tree_select)
        
        # Set column widths
        self.model_tree.setColumnWidth(0, 150)  # Model
        self.model_tree.setColumnWidth(1, 150)  # Parameter Size
        self.model_tree.setColumnWidth(2, 80)   # Size (GB)
        # Path column will stretch
        self.model_tree.header().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        
        model_layout.addWidget(self.model_tree)
        
        # Bottom section - Action tabs
        action_group = QGroupBox("Actions")
        self.main_layout.addWidget(action_group, 1)  # Smaller height ratio
        action_layout = QVBoxLayout(action_group)
        
        # Tabs for different actions
        self.tabs = QTabWidget()
        action_layout.addWidget(self.tabs)
        
        # Export tab
        export_widget = QWidget()
        self.tabs.addTab(export_widget, "Export Model")
        export_layout = QGridLayout(export_widget)
        
        # Export widgets
        export_layout.addWidget(QLabel("Select a model from the list above or enter model details:"), 
                               0, 0, 1, 3)
        
        export_layout.addWidget(QLabel("Model Name:"), 1, 0)
        self.export_model_edit = QLineEdit()
        export_layout.addWidget(self.export_model_edit, 1, 1)
        
        export_layout.addWidget(QLabel("Parameter Size:"), 2, 0)
        self.export_param_edit = QLineEdit()
        export_layout.addWidget(self.export_param_edit, 2, 1)
        
        export_layout.addWidget(QLabel("Output File:"), 3, 0)
        self.export_output_edit = QLineEdit()
        export_layout.addWidget(self.export_output_edit, 3, 1)
        
        export_browse_btn = QPushButton("Browse...")
        export_browse_btn.clicked.connect(self.browse_export_file)
        export_layout.addWidget(export_browse_btn, 3, 2)
        
        export_btn = QPushButton("Export Model")
        export_btn.clicked.connect(self.export_model)
        export_layout.addWidget(export_btn, 4, 0, 1, 3)
        
        # Import tab
        import_widget = QWidget()
        self.tabs.addTab(import_widget, "Import Model")
        import_layout = QGridLayout(import_widget)
        
        # Import widgets
        import_layout.addWidget(QLabel("Import a model from a .tar.gz file:"), 0, 0, 1, 3)
        
        import_layout.addWidget(QLabel("Archive File:"), 1, 0)
        self.import_file_edit = QLineEdit()
        import_layout.addWidget(self.import_file_edit, 1, 1)
        
        import_browse_btn = QPushButton("Browse...")
        import_browse_btn.clicked.connect(self.browse_import_file)
        import_layout.addWidget(import_browse_btn, 1, 2)
        
        import_layout.addWidget(QLabel("Custom Name (optional):"), 2, 0)
        self.import_name_edit = QLineEdit()
        import_layout.addWidget(self.import_name_edit, 2, 1)
        
        import_layout.addWidget(QLabel("Format: name:param_size"), 2, 2)
        
        import_btn = QPushButton("Import Model")
        import_btn.clicked.connect(self.import_model)
        import_layout.addWidget(import_btn, 3, 0, 1, 3)
        
        # Status bar and progress
        status_layout = QHBoxLayout()
        self.main_layout.addLayout(status_layout)
        
        self.status_label = QLabel("Ready")
        status_layout.addWidget(self.status_label)
        
        # Add a spinner label for visual feedback on operations
        self.spinner_index = 0
        self.spinner_symbols = ["⣾", "⣽", "⣻", "⢿", "⡿", "⣟", "⣯", "⣷"]
        self.spinner_label = QLabel("")
        self.spinner_label.setFont(QFont("Arial", 12))
        status_layout.addWidget(self.spinner_label)
        
        status_layout.addStretch()
        
        # Initialize operation tracking variables
        self.active_operation = False
        self.cancel_requested = False
        
        # Cancel button for long-running operations (initially hidden)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.cancel_operation)
        self.cancel_button.hide()  # Initially hidden
        status_layout.addWidget(self.cancel_button)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFixedWidth(200)
        status_layout.addWidget(self.progress_bar)
        
        # Timer for spinner animation
        self.spinner_timer = QTimer(self)
        self.spinner_timer.timeout.connect(self.update_spinner)
    
    def refresh_models(self):
        """Refresh the model list"""
        # Update status without triggering the operation UI state
        self.status_label.setText("Refreshing model list...")
        self.progress_bar.setValue(0)
        
        # Clear existing items
        self.model_tree.clear()
        
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
                item = QTreeWidgetItem(self.model_tree)
                item.setText(0, model_name)
                item.setText(1, variant["parameter_size"])
                item.setText(2, f"{variant['size_gb']:.2f}")
                item.setText(3, variant["path"])
                # Store model name and parameter size as data
                item.setData(0, Qt.ItemDataRole.UserRole, model_name)
                item.setData(1, Qt.ItemDataRole.UserRole, variant["parameter_size"])
        
        # Update status without triggering operation UI state
        self.status_label.setText("Model list refreshed")
        self.progress_bar.setValue(100)
        
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
    
    def on_tree_select(self):
        """Handle tree view selection"""
        selected_items = self.model_tree.selectedItems()
        if not selected_items:
            # Disable delete button if no selection
            self.delete_btn.setEnabled(False)
            return
            
        # Get selected item
        item = selected_items[0]
        
        # Get values from selected row
        model_name = item.text(0)
        param_size = item.text(1)
        
        # Fill export fields with selected model
        self.export_model_edit.setText(model_name)
        self.export_param_edit.setText(param_size)
        self.export_output_edit.setText(f"{model_name}-{param_size}.tar.gz")
        
        # Enable delete button when a model is selected
        self.delete_btn.setEnabled(True)
    
    def browse_export_file(self):
        """Browse for export file location"""
        try:
            model = self.export_model_edit.text()
            param = self.export_param_edit.text()
            default_name = f"{model}-{param}.tar.gz" if model and param else "model.tar.gz"
            
            # Use the save file dialog
            filename, _ = QFileDialog.getSaveFileName(
                self,
                "Save Model Archive",
                default_name,
                "TAR Archive (*.tar.gz);;All Files (*)"
            )
            
            # Only update if a valid filename was selected
            if filename:
                # Add .tar.gz extension if not present
                if not filename.endswith('.tar.gz'):
                    filename += '.tar.gz'
                self.export_output_edit.setText(filename)
        except Exception as e:
            print(f"Error in file dialog: {e}")
            QMessageBox.critical(self, "Error", f"Could not open file dialog: {e}")
    
    def browse_import_file(self):
        """Browse for import file"""
        try:
            # Use the open file dialog
            filename, _ = QFileDialog.getOpenFileName(
                self,
                "Select Model Archive",
                "",
                "TAR Archive (*.tar.gz);;All Files (*)"
            )
            
            # Only update if a valid filename was selected
            if filename:
                self.import_file_edit.setText(filename)
                
                # Try to extract model name from filename
                basename = os.path.basename(filename)
                if basename.endswith(".tar.gz"):
                    basename = basename[:-7]  # Remove .tar.gz
                    
                    # Try to parse model-param pattern
                    parts = basename.split("-")
                    if len(parts) >= 2:
                        model_name = "-".join(parts[:-1])  # Everything except last part
                        param_size = parts[-1]  # Last part
                        
                        self.import_name_edit.setText(f"{model_name}:{param_size}")
                    else:
                        # Handle simple case
                        self.import_name_edit.setText(f"{basename}:default")
        except Exception as e:
            print(f"Error in file dialog: {e}")
            QMessageBox.critical(self, "Error", f"Could not open file dialog: {e}")
    
    def export_model(self):
        """Export the selected model"""
        model_name = self.export_model_edit.text().strip()
        param_size = self.export_param_edit.text().strip()
        output_path = self.export_output_edit.text().strip()
        
        if not model_name or not param_size:
            QMessageBox.critical(self, "Error", "Please specify both model name and parameter size")
            return
            
        if not output_path:
            QMessageBox.critical(self, "Error", "Please specify an output file path")
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
            QMessageBox.critical(self, "Error", f"Could not find manifest for model {model_name}:{param_size}")
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
        archive_path = self.import_file_edit.text().strip()
        custom_name = self.import_name_edit.text().strip()
        
        if not archive_path:
            QMessageBox.critical(self, "Error", "Please select an archive file")
            return
            
        if not os.path.exists(archive_path):
            QMessageBox.critical(self, "Error", f"File not found: {archive_path}")
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
        selected_items = self.model_tree.selectedItems()
        if not selected_items:
            return
            
        # Get values from selected row
        item = selected_items[0]
        model_name = item.text(0)
        param_size = item.text(1)
        
        # Confirm deletion
        confirm = QMessageBox.question(
            self, 
            "Confirm Delete", 
            f"Are you sure you want to delete model {model_name}:{param_size}?\n\n"
            "This will delete the model manifest and associated blob files.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, 
            QMessageBox.StandardButton.No
        ) == QMessageBox.StandardButton.Yes
        
        if not confirm:
            return
            
        # First, clear the selection to avoid potential issues with deleted items
        self.model_tree.clearSelection()
        self.delete_btn.setEnabled(False)
        
        # Clear any export/import fields that might reference the model
        if self.export_model_edit.text() == model_name and self.export_param_edit.text() == param_size:
            self.export_model_edit.clear()
            self.export_param_edit.clear()
            self.export_output_edit.clear()
            
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
        self.status_label.setText(message)
        self.progress_bar.setValue(progress)
        
        # Handle spinner animation based on progress
        if progress < 100:
            # Operation in progress, show spinner and ensure cancel button is visible
            if not self.active_operation:
                self.start_operation()
        else:
            # Operation complete, hide spinner and cancel button
            self.end_operation()
    
    def update_spinner(self):
        """Update the spinner animation"""
        if not self.active_operation:
            return
            
        self.spinner_index = (self.spinner_index + 1) % len(self.spinner_symbols)
        self.spinner_label.setText(self.spinner_symbols[self.spinner_index])
    
    def start_operation(self):
        """Start a long-running operation"""
        self.active_operation = True
        self.cancel_requested = False
        
        # Show spinner
        self.spinner_label.setText(self.spinner_symbols[0])
        # Show cancel button
        self.cancel_button.show()
        # Start spinner animation
        self.spinner_timer.start(100)
        
        # Disable UI buttons in tabs
        for i in range(self.tabs.count()):
            tab = self.tabs.widget(i)
            for child in tab.findChildren(QPushButton):
                child.setEnabled(False)
                
        # Disable model tree interaction
        self.model_tree.setSelectionMode(QTreeWidget.SelectionMode.NoSelection)
        # Disable the refresh and delete buttons
        self.refresh_btn.setEnabled(False)
        self.delete_btn.setEnabled(False)
        
    def end_operation(self):
        """End a long-running operation"""
        self.active_operation = False
        # Hide spinner
        self.spinner_label.setText("")
        # Stop spinner timer
        self.spinner_timer.stop()
        # Hide cancel button
        self.cancel_button.hide()
        
        # Re-enable UI elements
        for i in range(self.tabs.count()):
            tab = self.tabs.widget(i)
            for child in tab.findChildren(QPushButton):
                child.setEnabled(True)
                
        # Re-enable model tree interaction
        self.model_tree.setSelectionMode(QTreeWidget.SelectionMode.SingleSelection)
        # Re-enable the refresh button
        self.refresh_btn.setEnabled(True)
        # The delete button state depends on selection, so we don't enable it here
        
    def cancel_operation(self):
        """Cancel the current operation"""
        if self.active_operation:
            self.cancel_requested = True
            self.status_label.setText("Cancelling operation...")
            self.progress_bar.setValue(0)
            # The thread will need to check cancel_requested periodically


if __name__ == "__main__":
    # Create the Qt Application
    qt_app = QApplication(sys.argv)
    # Create and show the GUI
    app = OllamaGUI()
    app.show()
    # Execute the application's main loop
    sys.exit(qt_app.exec())