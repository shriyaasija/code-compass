import os
from pathlib import Path

class CodeNode:
    """Represents a node in the code tree (folder, file, or function)"""

    # Define code vs non-code extensions
    CODE_EXTENSIONS = {'.py', '.js', '.java', '.cpp', '.c', '.go', '.rs', '.ts', '.jsx', '.tsx', '.rb', '.php', '.swift', '.kt', '.cs'}
    NON_CODE_EXTENSIONS = {'.md', '.txt', '.json', '.yaml', '.yml', '.toml', '.xml', '.csv', '.sql'}

    def __init__(self, title, node_id, node_type="file", path=None, start_line=None, end_line=None):
        self.title = title
        self.node_id = node_id
        self.type = node_type  # "folder", "file_py", "file_md", "function", "class"
        self.path = path
        self.start_line = start_line
        self.end_line = end_line
        self.text = None  # Only for non-code files
        self.summary = ""
        self.nodes = []  # Children nodes

    def set_type_from_extension(self):
        """Set type based on file extension"""
        if self.type == "file" and self.path:
            ext = Path(self.path).suffix.lower()

            if ext:
                # Convert .py to file_py, .md to file_md, etc.
                self.type = f"file{ext.replace('.', '_')}"
            else:
                self.type = "file"

    def should_store_text(self):
        """Determine if text should be stored in tree (only non-code files)"""
        if self.type.startswith("file_"):
            ext = self.type.replace("file_", ".")
            return ext in self.NON_CODE_EXTENSIONS
        return False

    def to_dict(self):
        """Convert to dictionary"""
        result = {
            "title": self.title,
            "node_id": self.node_id,
            "type": self.type,
            "path": self.path
        }

        # Add line numbers for functions/classes
        if self.start_line is not None:
            result["start_line"] = self.start_line
            result["end_line"] = self.end_line

        # Only include text for non-code files
        if self.text and self.should_store_text():
            result["text"] = self.text

        if self.summary:
            result["summary"] = self.summary

        if self.nodes:
            result["nodes"] = [child.to_dict() for child in self.nodes]

        return result


class TreeBuilder:
    """Builds directory tree with proper counter state"""

    def __init__(self, ignore_patterns=None):
        self.node_counter = 0  # Instance variable for counter
        self.ignore_patterns = ignore_patterns or [
            '.git', '__pycache__', 'node_modules', '.venv',
            'venv', 'build', 'dist', '.pytest_cache', '.DS_Store'
        ]

    def get_node_id(self):
        """Generate next node ID"""
        self.node_counter += 1
        return str(self.node_counter).zfill(4)

    def should_ignore(self, path):
        """Check if path should be ignored"""
        return any(pattern in path for pattern in self.ignore_patterns)

    def traverse_directory(self, dir_path):
        """Recursively traverse directory structure"""
        items = sorted(os.listdir(dir_path))
        nodes = []

        for item in items:
            item_path = os.path.join(dir_path, item)

            if self.should_ignore(item_path):
                continue

            if os.path.isdir(item_path):
                # Create folder node
                folder_node = CodeNode(
                    title=item,
                    node_id=self.get_node_id(),
                    node_type="folder",
                    path=item_path
                )
                # Recurse into subdirectories
                folder_node.nodes = self.traverse_directory(item_path)
                nodes.append(folder_node)

            elif os.path.isfile(item_path):
                # Create file node
                file_node = CodeNode(
                    title=item,
                    node_id=self.get_node_id(),
                    node_type="file",  # Will be set to file_py, file_md, etc.
                    path=item_path
                )
                # Set type based on extension
                file_node.set_type_from_extension()
                nodes.append(file_node)

        return nodes

    def build(self, repo_path):
        """Build complete directory tree"""
        # Reset counter for new tree
        self.node_counter = 0

        # Build root node
        root = CodeNode(
            title=os.path.basename(repo_path),
            node_id="0000",
            node_type="repository",
            path=repo_path
        )

        # Build children
        root.nodes = self.traverse_directory(repo_path)

        return root


def build_directory_tree(repo_path, ignore_patterns=None):
    """
    Convenience function to build hierarchical tree from repository directory structure
    Similar to PageIndex's process_not_toc() function
    """
    builder = TreeBuilder(ignore_patterns)
    return builder.build(repo_path)


# Test usage
if __name__ == "__main__":
    import json

    tree = build_directory_tree("/content/repos")
    print(json.dumps(tree.to_dict(), indent=2))
