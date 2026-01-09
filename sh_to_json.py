import argparse
import json
import shlex
from pathlib import Path


def _parse_assignment_line(line):
    line = line.strip()
    if not line or line.startswith('#') or '=' not in line:
        return None
    key, _, value = line.partition('=')
    key = key.strip()
    value = value.strip()
    if not key.isidentifier():
        return None
    if len(value) >= 2 and ((value[0] == "'" and value[-1] == "'") or (value[0] == '"' and value[-1] == '"')):
        value = value[1:-1]
    return key, value


def _parse_scalar(token):
    if isinstance(token, str) and token.startswith('$'):
        return token
    try:
        return int(token)
    except Exception:
        pass
    try:
        return float(token)
    except Exception:
        return token


def parse_sh_experiment(sh_path: Path):
    lines = sh_path.read_text(encoding='utf-8').splitlines()
    vars_map = {}
    for line in lines:
        kv = _parse_assignment_line(line)
        if kv:
            vars_map[kv[0]] = kv[1]

    cmd = None
    for line in reversed(lines):
        if 'python ' in line and (' run.py ' in line or ' restore.py ' in line):
            cmd = line.strip()
            break
    if not cmd:
        return None

    cmd = 'python' + cmd.split('python', 1)[1]
    tokens = shlex.split(cmd)
    entry = tokens[1]
    argv = tokens[2:]

    cfg = {'_entry': entry}
    i = 0
    while i < len(argv):
        arg = argv[i]
        if not arg.startswith('-'):
            i += 1
            continue

        key = arg.lstrip('-')
        if i + 1 < len(argv) and not argv[i + 1].startswith('-'):
            raw_value = argv[i + 1]
            if raw_value.startswith('$'):
                raw_value = vars_map.get(raw_value[1:], raw_value)
            cfg[key] = _parse_scalar(raw_value)
            i += 2
        else:
            cfg[key] = True
            i += 1

    for k in ('data', 'score_func'):
        if k in vars_map and k not in cfg:
            cfg[k] = vars_map[k]

    return cfg


def main():
    parser = argparse.ArgumentParser(description='Convert sh/*.sh experiment scripts into JSON config files.')
    parser.add_argument('--sh_dir', default='sh', help='Directory containing bash scripts (default: sh).')
    parser.add_argument('--out_dir', default='exp_configs', help='Output directory for JSON configs (default: exp_configs).')
    args = parser.parse_args()

    sh_dir = Path(args.sh_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for sh_path in sorted(sh_dir.glob('*.sh')):
        cfg = parse_sh_experiment(sh_path)
        if not cfg:
            continue
        out_path = out_dir / f'{sh_path.stem}.json'
        out_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
        count += 1

    print(f'Wrote {count} config(s) to: {out_dir}')


if __name__ == '__main__':
    main()
