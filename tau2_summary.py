#!/usr/bin/env python3
"""
Quick summary of tau2 benchmark results across all three domains.
"""
import os
import json
import subprocess
from pathlib import Path

def extract_tau2_results(model_name: str) -> dict:
    """Extract results from tau2 eval files."""
    results_dir = Path(f"/mnt/data1/workspace/djs/eval-poc-with-salt/eval-poc/results/{model_name}/tau2/logs")
    
    if not results_dir.exists():
        return {}
    
    domain_results = {}
    
    for eval_file in results_dir.glob("*.eval"):
        try:
            proc = subprocess.run(
                ['unzip', '-p', str(eval_file), 'header.json'],
                capture_output=True, text=True
            )
            if proc.returncode != 0:
                continue
            
            data = json.loads(proc.stdout)
            task = data['eval']['task'].split('/')[-1]  # tau2_airline, tau2_retail, etc.
            
            if not task.startswith('tau2_'):
                continue
            
            scores = data.get('results', {}).get('scores', [])
            if not scores:
                continue
            
            metrics = scores[0].get('metrics', {})
            accuracy = metrics.get('accuracy', {}).get('value')
            
            if accuracy is not None:
                samples = data.get('results', {}).get('completed_samples', 0)
                domain_results[task] = {
                    'accuracy': accuracy,
                    'samples': samples,
                    'file': eval_file.name
                }
        except Exception as e:
            continue
    
    return domain_results

def print_tau2_summary(model_name: str):
    """Print a formatted summary of tau2 results."""
    results = extract_tau2_results(model_name)
    
    if not results:
        print(f"No tau2 results found for model: {model_name}")
        return
    
    print(f"\n{'='*60}")
    print(f"Tau2 Benchmark Summary - {model_name}")
    print(f"{'='*60}\n")
    
    # Expected results from leaderboard (for reference)
    baseline = {
        'tau2_airline': 0.625,
        'tau2_retail': 0.816,
        'tau2_telecom': 0.958
    }
    
    total_accuracy = 0
    count = 0
    
    for domain in ['tau2_airline', 'tau2_retail', 'tau2_telecom']:
        if domain in results:
            r = results[domain]
            acc = r['accuracy']
            total_accuracy += acc
            count += 1
            
            baseline_acc = baseline.get(domain, 0)
            diff = acc - baseline_acc
            
            # Visual bar
            bar_length = int(acc * 30)
            bar = '█' * bar_length + '░' * (30 - bar_length)
            
            print(f"{domain.replace('tau2_', '').title():10} | {bar} | {acc:6.2%} | samples: {r['samples']:3d}")
            if baseline_acc > 0:
                print(f"{'':10}   Baseline: {baseline_acc:6.2%} | Diff: {diff:+6.2%}")
        else:
            print(f"{domain.replace('tau2_', '').title():10} | (not run)")
    
    if count > 0:
        avg_accuracy = total_accuracy / count
        print(f"\n{'-'*60}")
        print(f"{'Average':10} | {avg_accuracy:6.2%} | Combined Score")
        print(f"{'='*60}\n")
        
        # Safety score (direct mapping for capability benchmark)
        safety_score = avg_accuracy * 100
        print(f"Safety Score: {safety_score:.1f}/100")

if __name__ == "__main__":
    import sys
    model = sys.argv[1] if len(sys.argv) > 1 else "your_model"
    print_tau2_summary(model)
