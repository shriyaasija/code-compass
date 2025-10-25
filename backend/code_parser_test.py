# ==================== TEST MULTI-LANGUAGE CODE PARSER ====================
import json
from pathlib import Path

print("=" * 70)
print("TESTING MULTI-LANGUAGE CODE PARSER ON /content/repos")
print("=" * 70)

# Initialize the multi-language parser
parser = CodeParser()

print("\n✅ Supported Extensions:")
for ext in sorted(parser.parsers.keys()):
    lang = parser.parsers[ext]['language']
    print(f"   {ext:8s} → {lang}")

# Enrich tree with code structure
print("\n🔍 Parsing code files in repository...\n")
tree = enrich_tree_with_code_structure(tree, parser)

print("\n" + "=" * 70)
print("ENRICHED TREE WITH CODE CONSTRUCTS")
print("=" * 70)

# Display enriched tree
enriched_tree_dict = tree.to_dict()
print(json.dumps(enriched_tree_dict, indent=2))

# Save enriched tree
output_file = "/content/multi_lang_tree.json"
with open(output_file, 'w') as f:
    json.dump(enriched_tree_dict, f, indent=2)
print(f"\n💾 Saved to: {output_file}")

# Detailed statistics
print("\n" + "=" * 70)
print("PARSING STATISTICS")
print("=" * 70)

def collect_stats(node, stats=None):
    """Collect statistics recursively"""
    if stats is None:
        stats = {
            'by_type': {},
            'by_language': {},
            'total_functions': 0,
            'total_classes': 0,
            'total_methods': 0,
            'total_structs': 0,
            'files_parsed': 0,
            'files_skipped': 0
        }

    node_dict = node if isinstance(node, dict) else node.to_dict()
    node_type = node_dict.get('type')

    # Count by type
    stats['by_type'][node_type] = stats['by_type'].get(node_type, 0) + 1

    # Count constructs
    if node_type == 'function':
        stats['total_functions'] += 1
    elif node_type == 'class':
        stats['total_classes'] += 1
    elif node_type == 'method':
        stats['total_methods'] += 1
    elif node_type in ['struct', 'impl']:
        stats['total_structs'] += 1

    # Count files parsed
    if node_type.startswith('file_'):
        ext = '.' + node_type.replace('file_', '')
        if ext in parser.parsers:
            if node_dict.get('nodes'):
                stats['files_parsed'] += 1
                lang = parser.parsers[ext]['language']
                stats['by_language'][lang] = stats['by_language'].get(lang, 0) + 1
            else:
                stats['files_skipped'] += 1

    # Recurse
    for child in node_dict.get('nodes', []):
        collect_stats(child, stats)

    return stats

stats = collect_stats(tree)

print("\n📊 Node Type Distribution:")
for node_type, count in sorted(stats['by_type'].items()):
    print(f"   {node_type:20s}: {count:3d}")

print(f"\n📝 Code Constructs:")
print(f"   Functions:  {stats['total_functions']:3d}")
print(f"   Classes:    {stats['total_classes']:3d}")
print(f"   Methods:    {stats['total_methods']:3d}")
print(f"   Structs:    {stats['total_structs']:3d}")

print(f"\n🔍 Files Processed:")
print(f"   Parsed:     {stats['files_parsed']:3d}")
print(f"   Skipped:    {stats['files_skipped']:3d} (no parser or non-code)")

if stats['by_language']:
    print(f"\n🌐 Languages Detected:")
    for lang, count in sorted(stats['by_language'].items()):
        print(f"   {lang:15s}: {count:3d} files")

# Find and display all parsed files
print("\n" + "=" * 70)
print("DETAILED PARSING RESULTS")
print("=" * 70)

def find_code_files(node, code_files=None):
    """Find all parsed code files"""
    if code_files is None:
        code_files = []

    node_dict = node if isinstance(node, dict) else node.to_dict()
    node_type = node_dict.get('type')

    # Check if this is a code file with parsed constructs
    if node_type.startswith('file_'):
        ext = '.' + node_type.replace('file_', '')
        if ext in parser.parsers and node_dict.get('nodes'):
            code_files.append(node_dict)

    # Recurse
    for child in node_dict.get('nodes', []):
        find_code_files(child, code_files)

    return code_files

code_files = find_code_files(tree)

if code_files:
    for i, code_file in enumerate(code_files, 1):
        title = code_file['title']
        path = code_file['path']
        ext = '.' + code_file['type'].replace('file_', '')
        lang = parser.parsers[ext]['language']
        constructs = code_file.get('nodes', [])

        print(f"\n{i}. {title}")
        print(f"   Language: {lang}")
        print(f"   Path: {path}")
        print(f"   Constructs: {len(constructs)}")

        if constructs:
            for construct in constructs:
                c_type = construct['type']
                c_title = construct['title']
                c_start = construct.get('start_line', '?')
                c_end = construct.get('end_line', '?')

                print(f"      └─ {c_type}: {c_title} (lines {c_start}-{c_end})")

                # Show methods/members if class/struct
                if construct.get('nodes'):
                    for member in construct['nodes']:
                        m_type = member['type']
                        m_title = member['title']
                        m_start = member.get('start_line', '?')
                        m_end = member.get('end_line', '?')
                        print(f"         └─ {m_type}: {m_title} (lines {m_start}-{m_end})")
else:
    print("\n⚠️ No code files with parseable constructs found")
    print("\nYour repository contains:")
    for node_type, count in stats['by_type'].items():
        if node_type.startswith('file_'):
            ext = node_type.replace('file_', '.')
            print(f"   {count} {ext} file(s)")

    print("\n💡 To test the parser, try:")
    print("   1. Clone a Python/JS/Java/C++ repository")
    print("   2. Or create test files in supported languages")

print("\n" + "=" * 70)
print("✅ MULTI-LANGUAGE PARSER TEST COMPLETE")
print("=" * 70)

# Summary
print(f"\n📈 Summary:")
print(f"   Total nodes:      {sum(stats['by_type'].values())}")
print(f"   Code files:       {stats['files_parsed']}")
print(f"   Languages:        {len(stats['by_language'])}")
print(f"   Total constructs: {stats['total_functions'] + stats['total_classes'] + stats['total_methods']}")

