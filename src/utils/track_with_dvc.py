"""
Simple DVC tracking for automated splits
"""

import subprocess
import os
from pathlib import Path


def track_split(split_id: int) -> dict:
    """
    Track a split with DVC and Git
    
    Args:
        split_id: Split number (1-9)
        
    Returns:
        dict with status and commit hash
    """
    
    try:
        dvc_repo = Path("/app/dvc-tracking")
        split_source = Path("/app/data/automated_splits")
        
        # Find split directory
        pattern = f"split_{split_id:02d}_*"
        matches = list(split_source.glob(pattern))
        
        if not matches:
            return {
                "status": "error",
                "message": f"Split {split_id} not found in {split_source}"
            }
        
        split_dir = matches[0]
        split_name = split_dir.name
        
        print(f"[DVC] Tracking {split_name}...")
        
        # Change to DVC repo
        os.chdir(dvc_repo)
        
        # Copy split to DVC repo
        target_dir = dvc_repo / "data" / "automated_splits"
        target_dir.mkdir(parents=True, exist_ok=True)
        
        subprocess.run(
            ["cp", "-r", str(split_dir), str(target_dir)],
            check=True
        )
        
        # DVC add
        subprocess.run(
            ["dvc", "add", f"data/automated_splits/{split_name}"],
            check=True,
            capture_output=True
        )
        print(f"  [✓] DVC add completed")
        
        # Git add
        subprocess.run(
            ["git", "add", f"data/automated_splits/{split_name}.dvc", "data/automated_splits/.gitignore"],
            check=True
        )
        
        # Git commit
        subprocess.run(
            ["git", "commit", "-m", f"Track split {split_id}"],
            check=True,
            capture_output=True
        )
        
        # Get commit hash
        commit_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"]
        ).decode().strip()
        
        print(f"  [✓] Git commit: {commit_hash[:8]}")
        
        # DVC push
        result = subprocess.run(
            ["dvc", "push"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print(f"  [✓] DVC push completed")
        else:
            print(f"  [!] DVC push warning: {result.stderr}")
        
        # Git push
        subprocess.run(
            ["git", "push", "origin", "main"],
            check=True,
            capture_output=True
        )
        print(f"  [✓] Git push completed")

        
        # Git push to DagsHub
        result_dagshub = subprocess.run(
            ["git", "push", "dagshub", "main"],
            capture_output=True,
            text=True
        )
        if result_dagshub.returncode == 0:
            print(f"  [✓] Git push to DagsHub completed")
        else:
            print(f"  [!] Git push to DagsHub warning: {result_dagshub.stderr}")
        
        print(f"[✓] Split {split_id} tracked successfully!\n")
        
        return {
            "status": "success",
            "commit_hash": commit_hash,
            "split_id": split_id,
            "split_name": split_name
        }
        
    except subprocess.CalledProcessError as e:
        error_output = e.stderr if hasattr(e, 'stderr') and e.stderr else str(e)
        print(f"[ERROR] {error_output}")
        return {
            "status": "error",
            "message": error_output
        }
    except Exception as e:
        print(f"[ERROR] {str(e)}")
        return {
            "status": "error",
            "message": str(e)
        }


if __name__ == "__main__":
    # Test
    import sys
    if len(sys.argv) > 1:
        split_id = int(sys.argv[1])
        result = track_split(split_id)
        print(result)
