import argparse
import subprocess
import os
import shutil


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-b",
        "--branch",
        action="store",
        type=str,
        help="Name of new branch or exicting one",
        required=True,
    )
    parser.add_argument(
        "-v",
        "--ver",
        action="store",
        type=str,
        help="Verion of sdk in X.XX.XXX format",
        required=True,
    )
    parser.add_argument("-r", "--repo", action="store", type=str, required=True)
    parser.add_argument(
        "-p",
        "--path",
        action="store",
        type=str,
        help="Path to `requirements.txt` file; default `requirements.txt`",
        default="requirements.txt",
    )
    parser.add_argument(
        "--folder",
        action="store",
        type=str,
        help="Folder name where repo will be loaded; default `loaded_repo`",
        default="loaded_repo",
    )
    parser.add_argument(
        "-n",
        action="store_true",
        help="If true script will try to use `switch` command for new branch",
    )
    args = parser.parse_args()
    return args


def load_repo(repo: str, rfolder: str):
    print("loading...")
    subprocess.run(["git", "clone", repo, rfolder])


def create_and_change_branch(new_branch: str, rfolder: str, n_arg: bool):
    if n_arg:
        try:
            print("trying to change branch...")
            subprocess.run(["git", "switch", new_branch], cwd=f"./{rfolder}")
            return
        except Exception as e:
            print("can't switch")
    print("creating the new branch...")
    subprocess.run(["git", "checkout", "-b", new_branch], cwd=f"./{rfolder}")


def parse_reqs_and_change_version(path_to_reqs: str, rfolder: str, ver: str):
    print("changing `requirements.txt`...")
    path = os.path.join(rfolder, path_to_reqs)
    req_lines = []

    if not os.path.exists(path):
        raise ValueError("Wrong path to requirements.txt file.")

    with open(path, "r") as reqs:
        supervisely_added = False
        for line in reqs.readlines():
            line = line.strip()
            if line.startswith("supervisely") or line.startswith("# supervisely"):
                req_lines.append(f"supervisely=={ver}")
                supervisely_added = True
            elif line.startswith("git+https://github.com/supervisely/supervisely.git@"):
                req_lines.append(f"# {line}")
            else:
                req_lines.append(line)

        if not supervisely_added:
            req_lines.append(f"supervisely=={ver}")
    os.remove(path)

    with open(path, "w") as new_req:
        for line in req_lines:
            new_req.write(line + "\n")


def commit_and_push(ver: str, branch: str, rfolder: str):
    subprocess.run(["git", "add", "--all"], cwd=f"./{rfolder}")
    subprocess.run(["git", "commit", "-m", f"update sdk to {ver}"], cwd=f"./{rfolder}")
    subprocess.run(["git", "push", "origin", branch], cwd=f"./{rfolder}")


def clean_all(rfolder: str):
    shutil.rmtree(rfolder)


if __name__ == "__main__":
    args = parse_args()
    try:
        load_repo(args.repo, args.folder)
        create_and_change_branch(args.branch, args.folder, args.n)
        parse_reqs_and_change_version(args.path, args.folder, args.ver)
        commit_and_push(args.ver, args.branch, args.folder)
    finally:
        clean_all(args.folder)
        pass
