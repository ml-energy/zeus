"""Generate source code references pages."""

from pathlib import Path

import mkdocs_gen_files

REF_DIR = Path("reference")


nav = mkdocs_gen_files.Nav()

for path in sorted(Path("zeus").rglob("*.py")):
    # Path to the generated markdown file.
    doc_path = path.relative_to("zeus").with_suffix(".md")

    # Skip BSO server migration-related files.
    if str(doc_path).find("batch_size/migrations") != -1:
        continue

    full_doc_path = REF_DIR / doc_path
    module_path = path.with_suffix("")
    parts = tuple(module_path.parts)

    # zeus/run/__init__ can just be zeus/run.
    if parts[-1] == "__init__":
        parts = parts[:-1]
        doc_path = doc_path.with_name("index.md")
        full_doc_path = full_doc_path.with_name("index.md")
    # We currently don't have __main__ modules, but just in case.
    elif parts[-1] == "__main__":
        continue

    nav[parts] = doc_path.as_posix()

    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        identifier = ".".join(parts)
        fd.write(f"::: {identifier}\n")

with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as nav_file:
    nav_list = list(nav.build_literate_nav())
    nav_list[0] = "* [Source Code Reference](index.md)\n"
    nav_file.writelines(nav_list)
