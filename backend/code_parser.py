from tree_sitter import Language, Parser
from pathlib import Path

# Try importing all languages, handling different API versions
AVAILABLE_PARSERS = {}

# Python
try:
    import tree_sitter_python as tspython
    AVAILABLE_PARSERS['.py'] = ('python', tspython.language)
except ImportError:
    pass

# JavaScript
try:
    import tree_sitter_javascript as tsjavascript
    AVAILABLE_PARSERS['.js'] = ('javascript', tsjavascript.language)
    AVAILABLE_PARSERS['.jsx'] = ('javascript', tsjavascript.language)
except ImportError:
    pass

# TypeScript
try:
    import tree_sitter_typescript as tstypescript
    AVAILABLE_PARSERS['.ts'] = ('typescript', tstypescript.language_typescript)
    AVAILABLE_PARSERS['.tsx'] = ('typescript', tstypescript.language_tsx)
except ImportError:
    pass

# C++
try:
    import tree_sitter_cpp as tscpp
    AVAILABLE_PARSERS['.cpp'] = ('cpp', tscpp.language)
    AVAILABLE_PARSERS['.cc'] = ('cpp', tscpp.language)
    AVAILABLE_PARSERS['.cxx'] = ('cpp', tscpp.language)
    AVAILABLE_PARSERS['.hpp'] = ('cpp', tscpp.language)
except ImportError:
    pass

# C
try:
    import tree_sitter_c as tsc
    AVAILABLE_PARSERS['.c'] = ('c', tsc.language)
    AVAILABLE_PARSERS['.h'] = ('c', tsc.language)
except ImportError:
    pass

# Java
try:
    import tree_sitter_java as tsjava
    AVAILABLE_PARSERS['.java'] = ('java', tsjava.language)
except ImportError:
    pass

# Go
try:
    import tree_sitter_go as tsgo
    AVAILABLE_PARSERS['.go'] = ('go', tsgo.language)
except ImportError:
    pass

# Rust
try:
    import tree_sitter_rust as tsrust
    AVAILABLE_PARSERS['.rs'] = ('rust', tsrust.language)
except ImportError:
    pass

# Ruby
try:
    import tree_sitter_ruby as tsruby
    AVAILABLE_PARSERS['.rb'] = ('ruby', tsruby.language)
except ImportError:
    pass

# PHP - Skip if it has API issues
# try:
#     import tree_sitter_php as tsphp
#     AVAILABLE_PARSERS['.php'] = ('php', tsphp.language)
# except (ImportError, AttributeError):
#     pass


class CodeParser:
    """
    Multi-language code parser supporting 10+ programming languages
    Extracts functions, classes, methods from code files
    """

    # Language-specific node type mappings
    LANGUAGE_CONSTRUCTS = {
        'python': {
            'function': 'function_definition',
            'class': 'class_definition',
            'method': 'function_definition'
        },
        'javascript': {
            'function': ['function_declaration', 'function', 'arrow_function', 'method_definition'],
            'class': 'class_declaration',
            'method': 'method_definition'
        },
        'typescript': {
            'function': ['function_declaration', 'arrow_function', 'method_definition'],
            'class': 'class_declaration',
            'method': 'method_definition'
        },
        'cpp': {
            'function': 'function_definition',
            'class': 'class_specifier',
            'method': 'function_definition'
        },
        'c': {
            'function': 'function_definition',
            'struct': 'struct_specifier'
        },
        'java': {
            'function': 'method_declaration',
            'class': 'class_declaration',
            'method': 'method_declaration'
        },
        'go': {
            'function': 'function_declaration',
            'method': 'method_declaration',
            'struct': 'type_declaration'
        },
        'rust': {
            'function': 'function_item',
            'struct': 'struct_item',
            'impl': 'impl_item'
        },
        'ruby': {
            'function': 'method',
            'class': 'class',
            'module': 'module'
        }
    }

    def __init__(self):
        self.parsers = {}
        self.node_counter = 0

        print(f"🔧 Initializing parsers for {len(AVAILABLE_PARSERS)} extensions...")

        # Initialize all available parsers
        for ext, (lang_name, lang_func) in AVAILABLE_PARSERS.items():
            try:
                parser_info = self._create_parser(lang_func, lang_name)
                if parser_info:
                    self.parsers[ext] = parser_info
                    print(f"   ✅ {ext:6s} → {lang_name}")
            except Exception as e:
                print(f"   ⚠️  {ext:6s} → Failed: {e}")

        print(f"\n✅ CodeParser ready with {len(self.parsers)} language(s)")

    def _create_parser(self, language_func, lang_name):
        """Create parser for a specific language"""
        try:
            # Call the language function to get the language object
            lang_obj = language_func()
            LANG = Language(lang_obj)
            parser = Parser(LANG)
            return {'parser': parser, 'language': lang_name}
        except Exception as e:
            print(f"      Error creating parser for {lang_name}: {e}")
            return None

    def get_node_id(self, prefix="node"):
        """Generate unique node ID"""
        self.node_counter += 1
        return f"{prefix}_{self.node_counter}"

    def parse_file(self, file_path):
        """
        Extract functions and classes from a code file
        Returns list of CodeNode objects
        """
        ext = Path(file_path).suffix.lower()
        parser_info = self.parsers.get(ext)

        if not parser_info:
            return []

        parser = parser_info['parser']
        language = parser_info['language']

        try:
            with open(file_path, 'rb') as f:
                code = f.read()
        except Exception as e:
            print(f"⚠️ Could not read {file_path}: {e}")
            return []

        tree = parser.parse(code)
        root_node = tree.root_node

        constructs = self._extract_constructs(root_node, file_path, language)

        return constructs

    def _extract_constructs(self, root_node, file_path, language):
        """Extract language-specific constructs"""
        constructs = []
        construct_types = self.LANGUAGE_CONSTRUCTS.get(language, {})

        for child in root_node.children:
            node_type = child.type

            # Check for function definitions
            function_types = construct_types.get('function', [])
            if isinstance(function_types, str):
                function_types = [function_types]

            if node_type in function_types:
                constructs.append(CodeNode(
                    title=self._get_name(child),
                    node_id=self.get_node_id("func"),
                    node_type="function",
                    path=file_path,
                    start_line=child.start_point[0],
                    end_line=child.end_point[0]
                ))

            # Check for class definitions
            class_types = construct_types.get('class', [])
            if isinstance(class_types, str):
                class_types = [class_types]

            if node_type in class_types:
                class_node = CodeNode(
                    title=self._get_name(child),
                    node_id=self.get_node_id("class"),
                    node_type="class",
                    path=file_path,
                    start_line=child.start_point[0],
                    end_line=child.end_point[0]
                )
                # Extract methods
                class_node.nodes = self._extract_methods(child, file_path, language)
                constructs.append(class_node)

            # Check for struct/module/impl (language-specific)
            for construct_key in ['struct', 'module', 'impl']:
                struct_types = construct_types.get(construct_key, [])
                if isinstance(struct_types, str):
                    struct_types = [struct_types]

                if node_type in struct_types:
                    constructs.append(CodeNode(
                        title=self._get_name(child),
                        node_id=self.get_node_id(construct_key),
                        node_type=construct_key,
                        path=file_path,
                        start_line=child.start_point[0],
                        end_line=child.end_point[0]
                    ))

        return constructs

    # def _get_name(self, node):
    #     """Extract name from AST node (works across languages)"""
    #     # Try to find identifier child
    #     for child in node.children:
    #         if child.type in ['identifier', 'name', 'type_identifier']:
    #             return child.text.decode('utf8')

    #     # Fallback: check named children
    #     if hasattr(node, 'child_by_field_name'):
    #         name_node = node.child_by_field_name('name')
    #         if name_node:
    #             return name_node.text.decode('utf8')

    #     return "unknown"

    def _get_name(self, node):
        """Extract name from AST node (works across all languages)"""
        import re
        
        # Strategy 1: Try field-based name lookup (most reliable across languages)
        if hasattr(node, 'child_by_field_name'):
            # Try 'name' field (Python, JavaScript, TypeScript, Java, Go, Rust, Ruby, C++)
            name_node = node.child_by_field_name('name')
            if name_node:
                return name_node.text.decode('utf8')
            
            # Try 'declarator' field (C/C++/Java for complex declarations)
            declarator_node = node.child_by_field_name('declarator')
            if declarator_node:
                # Recursively extract identifier from any declarator structure
                def extract_identifier(decl):
                    if decl.type == 'identifier':
                        return decl.text.decode('utf8')
                    
                    # Check nested declarator field
                    if hasattr(decl, 'child_by_field_name'):
                        inner = decl.child_by_field_name('declarator')
                        if inner:
                            return extract_identifier(inner)
                        # Also try 'name' field within declarator
                        inner_name = decl.child_by_field_name('name')
                        if inner_name:
                            return inner_name.text.decode('utf8')
                    
                    # Search children for identifier or nested structures
                    for child in decl.children:
                        if child.type == 'identifier':
                            return child.text.decode('utf8')
                        # Recurse into declarator-like nodes
                        if 'declarator' in child.type or child.type in ['identifier', 'name']:
                            result = extract_identifier(child)
                            if result != "unknown":
                                return result
                    
                    return "unknown"
                
                result = extract_identifier(declarator_node)
                if result != "unknown":
                    return result
        
        # Strategy 2: Search direct children for identifier/name nodes
        # Works for most languages when field lookup isn't available
        for child in node.children:
            if child.type in ['identifier', 'name']:
                return child.text.decode('utf8')
        
        # Strategy 3: Regex extraction from node text as last resort
        # Handles edge cases and unusual structures across all languages
        try:
            node_text = node.text.decode('utf8')
            # Pattern 1: function_name(...) or method_name(...)
            match = re.search(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', node_text)
            if match:
                # Verify it's not a keyword
                potential_name = match.group(1)
                keywords = {'if', 'for', 'while', 'switch', 'return', 'class', 'def', 'function', 'const', 'let', 'var'}
                if potential_name not in keywords:
                    return potential_name
            
            # Pattern 2: class ClassName or struct StructName
            match = re.search(r'\b(?:class|struct|interface|module|impl)\s+([a-zA-Z_][a-zA-Z0-9_]*)', node_text)
            if match:
                return match.group(1)
            
            # Pattern 3: def method_name or function name (Ruby, Python)
            match = re.search(r'\b(?:def|fn)\s+([a-zA-Z_][a-zA-Z0-9_]*)', node_text)
            if match:
                return match.group(1)
        except:
            pass

        return "unknown"

    def _extract_methods(self, class_node, file_path, language):
        """Extract methods from a class"""
        methods = []
        construct_types = self.LANGUAGE_CONSTRUCTS.get(language, {})
        method_types = construct_types.get('method', [])
        if isinstance(method_types, str):
            method_types = [method_types]

        # Recursively search for methods
        def find_methods(node):
            if node.type in method_types:
                methods.append(CodeNode(
                    title=self._get_name(node),
                    node_id=self.get_node_id("method"),
                    node_type="method",
                    path=file_path,
                    start_line=node.start_point[0],
                    end_line=node.end_point[0]
                ))

            for child in node.children:
                find_methods(child)

        find_methods(class_node)
        return methods


def enrich_tree_with_code_structure(tree, parser):
    """Add function/class nodes to file nodes"""
    if tree.type.startswith("file_"):
        ext = Path(tree.path).suffix.lower()

        if ext in parser.parsers:
            print(f"🔍 Parsing {tree.title} ({parser.parsers[ext]['language']})...")
            constructs = parser.parse_file(tree.path)
            tree.nodes = constructs
            if constructs:
                print(f"   Found {len(constructs)} constructs")

    for child in tree.nodes:
        enrich_tree_with_code_structure(child, parser)

    return tree

