import yaml
import httpx
from pathlib import Path

current_dir = Path(__file__).parent.resolve()
config_path = current_dir / "weblinks.yaml"
weblinks_config = yaml.safe_load(config_path.read_text())

main_dir = current_dir.parent

data_dir = main_dir / '.data'
documentation_dir = data_dir / 'documentations'

documentation_dir.mkdir(exist_ok=True)

for name, link in weblinks_config.items():
    print(f"downloading LLM documenation for {name}")
    # link = "https://raw.githubusercontent.com/google/A2A/refs/heads/main/llms.txt"
    res = httpx.get(link)
    local_path = documentation_dir / f"{name}.md"
    local_path.write_bytes(res.content)
    print(f'Saved as {local_path.name}')

