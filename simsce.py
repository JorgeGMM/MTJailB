import os
import json
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from simcse import SimCSE
from sklearn.metrics.pairwise import cosine_similarity
from heapq import nlargest, nsmallest

# Ensure output directory exists
out_dir = 'simcse_outputs'
os.makedirs(out_dir, exist_ok=True)

plt.style.use('ggplot')  # clean, readable plot style

def load_and_group(paths):
    """
    Load multiple JSON files and combine them.
    Each file contains records with the new structure:
      - "conversation": dict of { "0": {"prompt": ..., "response": ..., "jailbreak_score": ...}, "1": ..., ... }
      - "is_harmful": bool
      - "source": str
      - "length": int
    Returns two lists of dicts (harmful, harmless), each entry:
      {
        'joined': str,              # full conversation joined
        'prompts_only': str,        # user prompts only
        'responses_only': str,      # model responses only
        'length': int,              # number of turns
        'turns': [list[str]],       # each turn's prompt as list of words
        'jailbreak_scores': [float], # jailbreak scores for each turn
        'source': str               # dataset source
      }
    """
    all_records = []
    
    # Load and combine all input files
    for path in paths:
        print(f"Loading {path}...")
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read().strip()
        if text.startswith('['):
            records = json.loads(text)
        else:
            records = [json.loads(line) for line in text.splitlines() if line.strip()]
        
        print(f"  Loaded {len(records)} conversations from {path}")
        all_records.extend(records)
    
    print(f"Total conversations loaded: {len(all_records)}")
    
    harmful, harmless = [], []
    for rec in all_records:
        L = rec['length']
        conv = rec['conversation']
        source = rec.get('source', 'unknown')
        
        # Extract different text representations
        prompts, responses, jailbreak_scores = [], [], []
        turn_words = []
        
        for i in range(L):
            turn_data = conv[str(i)]
            
            # Extract prompt and response
            prompt = turn_data.get('prompt', '').strip()
            response = turn_data.get('response', '').strip()
            jb_score = turn_data.get('jailbreak_score', 0.0)
            
            prompts.append(prompt)
            responses.append(response)
            jailbreak_scores.append(jb_score)
            turn_words.append(prompt.split())
        
        # Create different text representations
        joined_full = '  '.join([f"User: {p}  Assistant: {r}" for p, r in zip(prompts, responses)])
        joined_prompts = '  '.join(prompts)
        joined_responses = '  '.join(responses)
        
        entry = {
            'joined': joined_full,
            'prompts_only': joined_prompts,
            'responses_only': joined_responses,
            'length': L,
            'turns': turn_words,
            'jailbreak_scores': jailbreak_scores,
            'source': source
        }
        
        if rec['is_harmful']:
            harmful.append(entry)
        else:
            harmless.append(entry)
    
    print(f"Harmful conversations: {len(harmful)}")
    print(f"Harmless conversations: {len(harmless)}")
    return harmful, harmless


def embed_texts(texts, model, batch_size=64):
    """
    Encode a list of sentences into embeddings (n × d).
    """
    return np.array(model.encode(texts, batch_size=batch_size))


def analyze_jailbreak_scores(harmful_conversations):
    """
    Analyze the distribution and patterns of jailbreak scores.
    Cleans negative scores by setting them to 0 (as negative values are tracking artifacts).
    """
    all_scores = []
    scores_by_turn = defaultdict(list)
    scores_by_length = defaultdict(list)
    
    # Track cleaning statistics
    original_negative_count = 0
    total_score_count = 0
    
    for conv in harmful_conversations:
        scores = conv['jailbreak_scores']
        length = conv['length']
        
        # Clean negative scores (set to 0) and track changes
        cleaned_scores = []
        for score in scores:
            total_score_count += 1
            if score < 0:
                original_negative_count += 1
                cleaned_scores.append(0.0)
            else:
                cleaned_scores.append(score)
        
        # Update conversation with cleaned scores
        conv['jailbreak_scores'] = cleaned_scores
        
        for turn_idx, score in enumerate(cleaned_scores):
            all_scores.append(score)
            scores_by_turn[turn_idx].append(score)
            scores_by_length[length].append(score)
    
    analysis = {
        'data_cleaning': {
            'total_scores': total_score_count,
            'negative_scores_found': original_negative_count,
            'negative_score_percentage': float(original_negative_count / total_score_count * 100) if total_score_count > 0 else 0.0,
            'cleaned_to_zero': original_negative_count
        },
        'overall': {
            'mean': float(np.mean(all_scores)),
            'median': float(np.median(all_scores)),
            'std': float(np.std(all_scores)),
            'min': float(np.min(all_scores)),
            'max': float(np.max(all_scores)),
            'zero_score_count': int(np.sum(np.array(all_scores) == 0.0)),
            'zero_score_percentage': float(np.sum(np.array(all_scores) == 0.0) / len(all_scores) * 100) if all_scores else 0.0
        },
        'by_turn': {
            turn: {
                'mean': float(np.mean(scores)),
                'median': float(np.median(scores)),
                'std': float(np.std(scores)),
                'count': len(scores),
                'zero_count': int(np.sum(np.array(scores) == 0.0)),
                'success_rate': float(np.sum(np.array(scores) > 0.5) / len(scores) * 100) if scores else 0.0
            }
            for turn, scores in scores_by_turn.items()
        },
        'by_length': {
            length: {
                'mean': float(np.mean(scores)),
                'median': float(np.median(scores)),
                'std': float(np.std(scores)),
                'count': len(scores),
                'zero_count': int(np.sum(np.array(scores) == 0.0)),
                'success_rate': float(np.sum(np.array(scores) > 0.5) / len(scores) * 100) if scores else 0.0
            }
            for length, scores in scores_by_length.items()
        }
    }
    
    # Print cleaning summary
    print(f"Data cleaning summary:")
    print(f"  - Total jailbreak scores: {total_score_count}")
    print(f"  - Negative scores found: {original_negative_count} ({original_negative_count/total_score_count*100:.1f}%)")
    print(f"  - Negative scores cleaned to 0.0: {original_negative_count}")
    
    return analysis

def sample_extremes(emb_h, texts_h, emb_n, texts_n, L, k=3):
    """
    Print top-k most and bottom-k least similar pairs for convos of length L.
    """
    if len(emb_h) == 0 or len(emb_n) == 0:
        print(f"No data for length {L}")
        return
        
    sims = cosine_similarity(emb_h, emb_n).ravel()
    m = emb_n.shape[0]
    print(f"\n-- Length = {L}, extremes (k={k}) --")
    print("  MOST similar:")
    for score, idx in nlargest(min(k, len(sims)), zip(sims, range(len(sims)))):
        i, j = divmod(idx, m)
        print(f"    +{score:.3f}  H[{i}]: {texts_h[i][:100]!r}...  ↔  N[{j}]: {texts_n[j][:100]!r}...")
    print("  LEAST similar:")
    for score, idx in nsmallest(min(k, len(sims)), zip(sims, range(len(sims)))):
        i, j = divmod(idx, m)
        print(f"    –{score:.3f}  H[{i}]: {texts_h[i][:100]!r}...  ↔  N[{j}]: {texts_n[j][:100]!r}...")


def analyze_semantic_similarity(harmful, harmless, model, text_type='joined'):
    """
    Perform comprehensive semantic similarity analysis using SimCSE.
    
    Args:
        harmful, harmless: Lists of conversation dictionaries
        model: SimCSE model
        text_type: 'joined', 'prompts_only', or 'responses_only'
    """
    print(f"\n=== Semantic Analysis using {text_type} ===")
    
    # Extract texts based on type
    texts_h = [r[text_type] for r in harmful]
    texts_n = [r[text_type] for r in harmless]
    
    # Embed texts
    print("Computing embeddings...")
    emb_h = embed_texts(texts_h, model, batch_size=64)
    emb_n = embed_texts(texts_n, model, batch_size=64)
    
    # Compute semantic metrics
    print("Computing similarity metrics...")
    
    # 1. Prototype similarity (centroids)
    proto_sim = float(cosine_similarity(emb_h.mean(axis=0).reshape(1,-1),
                                        emb_n.mean(axis=0).reshape(1,-1))[0,0])
    
    # 2. Global cross-class similarity
    global_cross = float(cosine_similarity(emb_h, emb_n).mean())
    
    # 3. Intra-class similarities
    sim_h = cosine_similarity(emb_h); n_h = sim_h.shape[0]
    sim_n = cosine_similarity(emb_n); n_n = sim_n.shape[0]
    intra_h = float((sim_h.sum() - n_h) / (n_h * (n_h - 1))) if n_h > 1 else 0.0
    intra_n = float((sim_n.sum() - n_n) / (n_n * (n_n - 1))) if n_n > 1 else 0.0
    
    # 4. Cross-similarity by conversation length
    by_len_h = defaultdict(list)
    by_len_n = defaultdict(list)
    for r, vec in zip(harmful, emb_h): 
        by_len_h[r['length']].append(vec)
    for r, vec in zip(harmless, emb_n): 
        by_len_n[r['length']].append(vec)
    
    cross_by_len = {}
    for L in sorted(set(by_len_h) & set(by_len_n)):
        if len(by_len_h[L]) > 0 and len(by_len_n[L]) > 0:
            cross_by_len[L] = float(cosine_similarity(np.vstack(by_len_h[L]),
                                                     np.vstack(by_len_n[L])).mean())
    
    # 5. Cross-similarity by source dataset
    by_source_h = defaultdict(list)
    by_source_n = defaultdict(list)
    for r, vec in zip(harmful, emb_h):
        by_source_h[r['source']].append(vec)
    for r, vec in zip(harmless, emb_n):
        by_source_n[r['source']].append(vec)
    
    cross_by_source = {}
    for source in sorted(set(by_source_h) & set(by_source_n)):
        if len(by_source_h[source]) > 0 and len(by_source_n[source]) > 0:
            cross_by_source[source] = float(cosine_similarity(np.vstack(by_source_h[source]),
                                                             np.vstack(by_source_n[source])).mean())
    
    results = {
        'prototype_similarity': proto_sim,
        'global_cross_similarity': global_cross,
        'intra_harmful_similarity': intra_h,
        'intra_harmless_similarity': intra_n,
        'cross_similarity_by_length': cross_by_len,
        'cross_similarity_by_source': cross_by_source,
        'separability_index': intra_h + intra_n - 2 * global_cross,  # Higher = more separable
        'cluster_quality': (intra_h + intra_n) / 2,  # Higher = tighter clusters
    }
    
    print(f"Prototype (centroid) similarity: {proto_sim:.4f}")
    print(f"Global cross-class similarity: {global_cross:.4f}")
    print(f"Intra-class harmful: {intra_h:.4f}")
    print(f"Intra-class harmless: {intra_n:.4f}")
    print(f"Separability index: {results['separability_index']:.4f}")
    print(f"Cluster quality: {results['cluster_quality']:.4f}")
    
    return results, (emb_h, emb_n, texts_h, texts_n, by_len_h, by_len_n)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Analyze semantic/length bias in harmful vs. harmless convos'
    )
    parser.add_argument('json_paths', nargs='+', help='Paths to JSON files')
    parser.add_argument('--model', default='princeton-nlp/unsup-simcse-bert-base-uncased',
                        help='SimCSE model')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size')
    parser.add_argument('--peek', action='store_true',
                        help='Print sample extremes')
    parser.add_argument('--sample_length', type=int, default=None,
                        help='Sample extremes for this length')
    parser.add_argument('--k', type=int, default=3,
                        help='Number of extremes to sample')
    args = parser.parse_args()

    # Dataset name for filenames
    base = "_".join([os.path.splitext(os.path.basename(p))[0] for p in args.json_paths])
    if len(base) > 50:  # Truncate if too long
        base = f"combined_{len(args.json_paths)}_datasets"

    # Load & group from multiple files
    harmful, harmless = load_and_group(args.json_paths)

    # Load SimCSE model
    print(f"Loading SimCSE model: {args.model}")
    model = SimCSE(args.model)

    # Analyze jailbreak scores for harmful conversations
    print("\n=== Jailbreak Score Analysis ===")
    jb_analysis = analyze_jailbreak_scores(harmful)
    
    # Perform semantic analysis for different text representations
    analysis_results = {}
    embeddings_cache = {}
    
    for text_type in ['joined', 'prompts_only', 'responses_only']:
        semantic_results, embeddings_data = analyze_semantic_similarity(
            harmful, harmless, model, text_type
        )
        analysis_results[text_type] = semantic_results
        embeddings_cache[text_type] = embeddings_data

    # Conversation length stats (using word counts)
    conv_lengths = {
        'harmful': [len(r['joined'].split()) for r in harmful],
        'harmless': [len(r['joined'].split()) for r in harmless]
    }
    conv_stats = {
        lbl: {
            'count': len(vals),
            'mean': float(np.mean(vals)),
            'median': float(np.median(vals)),
            'std': float(np.std(vals)),
            'min': float(np.min(vals)),
            'max': float(np.max(vals))
        }
        for lbl, vals in conv_lengths.items()
    }

    # Turn-level stats (prompt length by turn position)
    turn_stats = {}
    for lbl, recs in [('harmful', harmful), ('harmless', harmless)]:
        tdict = defaultdict(list)
        for r in recs:
            for idx, turn in enumerate(r['turns'], start=1):
                tdict[idx].append(len(turn))
        turn_stats[lbl] = {
            i: {
                'mean': float(np.mean(lengths)),
                'median': float(np.median(lengths)),
                'std': float(np.std(lengths)),
                'count': len(lengths)
            }
            for i, lengths in tdict.items()
        }

    # Source distribution analysis
    source_stats = {
        'harmful': defaultdict(int),
        'harmless': defaultdict(int)
    }
    for r in harmful:
        source_stats['harmful'][r['source']] += 1
    for r in harmless:
        source_stats['harmless'][r['source']] += 1

    # Compile comprehensive analysis
    full_analysis = {
        'semantic_analysis': analysis_results,
        'jailbreak_scores': jb_analysis,
        'length_analysis': {
            'conversation': conv_stats,
            'turn': turn_stats
        },
        'source_distribution': dict(source_stats),
        'dataset_info': {
            'total_harmful': len(harmful),
            'total_harmless': len(harmless),
            'input_files': args.json_paths
        }
    }

    # Save comprehensive analysis
    output_file = os.path.join(out_dir, f'{base}_comprehensive_analysis.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(full_analysis, f, indent=2)
    print(f"\nSaved comprehensive analysis to: {output_file}")

    # Optional: sample extremes for a specific length
    if args.peek and args.sample_length:
        L = args.sample_length
        emb_h, emb_n, texts_h, texts_n, by_len_h, by_len_n = embeddings_cache['joined']
        if L in by_len_h and L in by_len_n:
            sample_extremes(
                np.vstack(by_len_h[L]),
                [r['joined'] for r in harmful if r['length'] == L],
                np.vstack(by_len_n[L]),
                [r['joined'] for r in harmless if r['length'] == L],
                L, args.k
            )

    # VISUALIZATION
    print("\nGenerating visualizations...")
    
    # 1. Conversation length histograms
    all_vals = conv_lengths['harmful'] + conv_lengths['harmless']
    bins = np.histogram_bin_edges(all_vals, bins=20)
    counts_h, _ = np.histogram(conv_lengths['harmful'], bins=bins)
    counts_n, _ = np.histogram(conv_lengths['harmless'], bins=bins)
    ymax = max(counts_h.max(), counts_n.max()) * 1.1
    
    for lbl in ['harmful', 'harmless']:
        data = conv_lengths[lbl]
        plt.figure(figsize=(10, 6))
        plt.hist(data, bins=bins, alpha=0.7, edgecolor='black')
        plt.ylim(0, ymax)
        plt.title(f"Words per conversation ({lbl})")
        plt.xlabel("Words")
        plt.ylabel("Frequency")
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(out_dir, f'{base}_hist_conv_{lbl}.png'), dpi=300, bbox_inches='tight')
        plt.close()

    # 2. Average conversation length comparison
    plt.figure(figsize=(8, 6))
    classes = list(conv_stats.keys())
    means = [conv_stats[c]['mean'] for c in classes]
    stds = [conv_stats[c]['std'] for c in classes]
    plt.bar(classes, means, yerr=stds, capsize=5, alpha=0.7)
    plt.ylim(0, max(means) * 1.2)
    plt.title("Average conversation length by class")
    plt.ylabel("Mean words (± std)")
    plt.savefig(os.path.join(out_dir, f'{base}_bar_conv_mean.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Turn-level analysis
    all_turn_means = [v['mean'] for stats in turn_stats.values() for v in stats.values()]
    y_max_turn = max(all_turn_means) * 1.1 if all_turn_means else 100
    
    for lbl, stats in turn_stats.items():
        if not stats:
            continue
        idxs = sorted(stats.keys())
        vals = [stats[i]['mean'] for i in idxs]
        errs = [stats[i]['std'] for i in idxs]
        
        plt.figure(figsize=(10, 6))
        plt.errorbar(idxs, vals, yerr=errs, marker='o', capsize=5)
        plt.ylim(0, y_max_turn)
        plt.title(f"Average words per turn index ({lbl})")
        plt.xlabel("Turn index")
        plt.ylabel("Mean words (± std)")
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(out_dir, f'{base}_line_turn_{lbl}.png'), dpi=300, bbox_inches='tight')
        plt.close()

    # 4. Semantic similarity heatmap
    text_types = ['joined', 'prompts_only', 'responses_only']
    metrics = ['prototype_similarity', 'global_cross_similarity', 'intra_harmful_similarity', 'intra_harmless_similarity']
    
    fig, ax = plt.subplots(figsize=(12, 8))
    data_matrix = np.array([[analysis_results[tt][metric] for metric in metrics] for tt in text_types])
    
    im = ax.imshow(data_matrix, cmap='RdYlBu_r', aspect='auto')
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels([m.replace('_', '\n') for m in metrics], rotation=45, ha='right')
    ax.set_yticks(range(len(text_types)))
    ax.set_yticklabels([tt.replace('_', '\n') for tt in text_types])
    
    # Add text annotations
    for i in range(len(text_types)):
        for j in range(len(metrics)):
            ax.text(j, i, f'{data_matrix[i, j]:.3f}', ha='center', va='center', fontweight='bold')
    
    plt.colorbar(im, ax=ax, label='Cosine Similarity')
    plt.title('Semantic Similarity Analysis Across Text Types')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'{base}_semantic_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Analysis complete! Check {out_dir}/ for outputs.")
    print(f"Key findings:")
    print(f"  - Harmful conversations: {len(harmful)}")
    print(f"  - Harmless conversations: {len(harmless)}")
    print(f"  - Prototype similarity (full conversation): {analysis_results['joined']['prototype_similarity']:.4f}")
    print(f"  - Separability index (full conversation): {analysis_results['joined']['separability_index']:.4f}")

    # 5. Jailbreak score analysis visualizations
    if jb_analysis['overall']['max'] > 0:  # Only if we have jailbreak scores
        
        # 5a. Overall jailbreak score distribution
        all_jb_scores = []
        for conv in harmful:
            all_jb_scores.extend(conv['jailbreak_scores'])
        
        plt.figure(figsize=(10, 6))
        plt.hist(all_jb_scores, bins=20, alpha=0.7, edgecolor='black')
        plt.axvline(x=0.5, color='red', linestyle='--', label='Success threshold (0.5)')
        plt.axvline(x=np.mean(all_jb_scores), color='orange', linestyle='-', label=f'Mean ({np.mean(all_jb_scores):.3f})')
        plt.title("Jailbreak Score Distribution (Cleaned)")
        plt.xlabel("Jailbreak Score")
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(out_dir, f'{base}_jailbreak_score_dist.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 5b. Success rate by turn
        turn_data = jb_analysis['by_turn']
        if turn_data:
            turns = sorted(turn_data.keys())
            success_rates = [turn_data[t]['success_rate'] for t in turns]
            
            plt.figure(figsize=(10, 6))
            plt.bar(turns, success_rates, alpha=0.7)
            plt.title("Jailbreak Success Rate by Turn")
            plt.xlabel("Turn Index")
            plt.ylabel("Success Rate (%)")
            plt.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for turn, rate in zip(turns, success_rates):
                plt.text(turn, rate + 1, f'{rate:.1f}%', ha='center', va='bottom')
            
            plt.savefig(os.path.join(out_dir, f'{base}_success_by_turn.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        # 5c. Success rate by conversation length
        length_data = jb_analysis['by_length']
        if length_data:
            lengths = sorted(length_data.keys())
            length_success_rates = [length_data[l]['success_rate'] for l in lengths]
            
            plt.figure(figsize=(10, 6))
            plt.bar(lengths, length_success_rates, alpha=0.7)
            plt.title("Jailbreak Success Rate by Conversation Length")
            plt.xlabel("Conversation Length (turns)")
            plt.ylabel("Success Rate (%)")
            plt.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for length, rate in zip(lengths, length_success_rates):
                plt.text(length, rate + 1, f'{rate:.1f}%', ha='center', va='bottom')
            
            plt.savefig(os.path.join(out_dir, f'{base}_success_by_length.png'), dpi=300, bbox_inches='tight')
            plt.close()