# LLMdex — Atlas visuel des modèles LLM locaux

## Vision

Un "Pokédex" pour les modèles de langage locaux. Un script Python lit les
fichiers GGUF (via Ollama ou directement), extrait métadonnées et
statistiques des poids, et génère des planches visuelles HTML — une
empreinte unique par modèle, comme un herbier.

## Architecture

```
GGUF → extract.py → JSON → build.py → index.html
```

Trois composants indépendants reliés par un format JSON intermédiaire :

- `extract.py` — Lit un GGUF, produit un JSON dans `data/`.
  Accepte un nom de modèle Ollama ou un chemin GGUF direct.
- `build.py` — Lit tous les JSON de `data/`, les injecte dans
  `template.html` comme variable JS, produit `dist/index.html`.
- `template.html` — Page atlas (CSS+JS inline), zéro dépendances externes.

### Arborescence cible

```
LLMdex/
├── extract.py              # GGUF → JSON
├── build.py                # Assemble data/ dans le template
├── template.html           # Page atlas source
├── data/                   # Un JSON par modèle extrait
│   └── qwen3.6-coding.json
├── dist/                   # Fichier final déployable
│   └── index.html
└── CLAUDE.md
```

### Flow d'utilisation

```bash
python extract.py qwen3.6-coding        # Un modèle Ollama
python extract.py --all                  # Tous les modèles installés
python build.py                          # Assemble → dist/index.html
open dist/index.html                     # Ou déployer
```

## Dépendances

Python : `gguf`, `numpy`. Rien d'autre côté extraction.
Front : zéro dépendances. Vanilla JS, Canvas pour le spectre, SVG pour
le diagramme architectural. Dark/light theme natif.

```bash
pip install gguf numpy
```

## Extraction : localiser les GGUF Ollama

Ollama stocke les modèles dans `~/.ollama/models/`. Le chemin vers le
blob GGUF se résout ainsi :

1. Lire le manifeste JSON :
   `~/.ollama/models/manifests/registry.ollama.ai/library/{nom}/{tag}`
   (tag = `latest` par défaut)
2. Trouver la layer de type `application/vnd.ollama.image.model`
3. Suivre son `digest` vers :
   `~/.ollama/models/blobs/sha256-{hash}`

Ce blob est le fichier GGUF.

## Lecture GGUF en Python

### API gguf (gguf-py de llama.cpp)

La lib officielle expose `GGUFReader` qui utilise `numpy.memmap` :

```python
from gguf import GGUFReader

reader = GGUFReader("/path/to/model.gguf")

# Métadonnées : reader.fields est un OrderedDict de ReaderField
for key, field in reader.fields.items():
    print(key, field.data)

# Tenseurs : itérer sur reader.tensors (list de ReaderTensor)
for tensor in reader.tensors:
    print(tensor.name, tensor.shape, tensor.tensor_type)
    # tensor.data donne un numpy array (mmap, pas chargé en RAM)
```

Alternative plus simple : `gguf-parser` (pip install gguf-parser) qui
expose `GGUFParser` avec parse()/print().

### Clés de métadonnées standard

Les clés sont préfixées par l'architecture (ex: `llama.`, `qwen2moe.`).
La clé `general.architecture` donne le préfixe.

```
general.architecture          str     "llama", "qwen2moe", "gemma", ...
general.name                  str     Nom lisible du modèle
general.file_type             uint32  Type de quantification globale

{arch}.block_count            uint32  Nombre de couches transformer
{arch}.context_length         uint32  Longueur de contexte max
{arch}.embedding_length       uint32  Dimension des embeddings
{arch}.feed_forward_length    uint32  Dimension feed-forward
{arch}.attention.head_count   uint32  Têtes d'attention
{arch}.attention.head_count_kv uint32 Têtes KV (GQA si ≠ head_count)
{arch}.attention.layer_norm_rms_epsilon float32
{arch}.vocab_size             uint32  Taille du vocabulaire
{arch}.expert_count           uint32  Nombre d'experts (MoE)
{arch}.expert_used_count      uint32  Experts actifs par token (MoE)
```

## Convention de nommage des tenseurs GGUF

Tous les noms suivent le pattern `blk.{N}.{composant}.weight` (ou `.bias`).
Le format GGUF normalise les noms quelle que soit l'architecture source.

### Tenseurs globaux (hors blocs)

```
token_embd.weight             Embedding des tokens
output.weight                 Projection de sortie (lm_head)
output_norm.weight            LayerNorm finale
```

### Tenseurs par bloc transformer (blk.{N}.*)

```
# Attention
blk.{N}.attn_q.weight        Query projection
blk.{N}.attn_k.weight        Key projection
blk.{N}.attn_v.weight        Value projection
blk.{N}.attn_output.weight   Output projection
blk.{N}.attn_norm.weight     Pre-attention LayerNorm

# Feed-forward
blk.{N}.ffn_gate.weight      Gate projection (SwiGLU)
blk.{N}.ffn_up.weight        Up projection
blk.{N}.ffn_down.weight      Down projection
blk.{N}.ffn_norm.weight      Pre-FFN LayerNorm
```

### Tenseurs MoE additionnels (blk.{N}.*)

Pour les modèles mixture-of-experts (comme qwen3.6-coding) :

```
blk.{N}.ffn_gate_inp.weight           Router / gating network
blk.{N}.ffn_gate_exps.weight          Gate projections (tous experts empilés)
blk.{N}.ffn_up_exps.weight            Up projections (tous experts empilés)
blk.{N}.ffn_down_exps.weight          Down projections (tous experts empilés)
# OU par expert individuel :
blk.{N}.ffn_gate.{E}.weight           Gate projection expert E
blk.{N}.ffn_up.{E}.weight             Up projection expert E
blk.{N}.ffn_down.{E}.weight           Down projection expert E
```

### Mapping de normalisation pour extract.py

Le script doit normaliser les noms de tenseurs vers un schéma unifié
pour la visualisation. Mapping cible :

```python
COMPONENT_MAP = {
    "attn_q":      "attn.q",
    "attn_k":      "attn.k",
    "attn_v":      "attn.v",
    "attn_output": "attn.o",
    "ffn_gate":    "ff.gate",
    "ffn_up":      "ff.up",
    "ffn_down":    "ff.down",
    "attn_norm":   "norm.attn",
    "ffn_norm":    "norm.ffn",
}
```

## Modèle de données JSON

Chaque modèle produit un JSON avec deux blocs.

### Bloc `meta` — Identité

```json
{
  "name": "qwen3.6-coding",
  "architecture": "qwen2moe",
  "params_total": 35000000000,
  "params_active": 3000000000,
  "quantization": "mxfp8",
  "file_size": 37000000000,
  "n_layers": 64,
  "n_heads": 40,
  "n_kv_heads": 8,
  "d_embed": 2560,
  "d_ff": 18944,
  "context_length": 16384,
  "vocab_size": 151936,
  "n_experts": 128,
  "n_experts_active": 8
}
```

### Bloc `tensor_stats` — Statistiques par tenseur

```json
{
  "tensor_stats": [
    {
      "layer": 0,
      "component": "attn.q",
      "shape": [2560, 2560],
      "dtype": "Q8_0",
      "n_params": 6553600,
      "mean": 0.0012,
      "std": 0.0234,
      "norm": 1.872,
      "min": -0.156,
      "max": 0.148,
      "sparsity": 0.02
    }
  ]
}
```

## Design visuel : la planche

Chaque modèle est une planche composée de trois zones.

### Zone 1 — En-tête d'identité

Nom en display, famille architecturale, stats clés en grille compacte :
paramètres (total / actifs si MoE), quantification, contexte, taille disque.

### Zone 2 — Diagramme architectural

SVG généré en JS. Représentation schématique des blocs transformer avec
proportions visibles (profond/étroit vs large/court). Montre le ratio GQA
et la structure MoE. Interactif : hover montre les dimensions.

### Zone 3 — Spectre des poids

Canvas 2D : colonnes = couches (0→n), lignes = composants. Encodage :
- Teinte : type de composant (bleu=attention, rouge=feed-forward,
  violet=normes, orange=embeddings)
- Opacité : norme L2 relative (normalisée sur le modèle)

Deux modèles de la même famille partagent un motif similaire mais décalé.
Un fine-tune altère les couches hautes.

## Étapes d'implémentation

### Étape 1 — Extraction des métadonnées
Résoudre le chemin GGUF depuis Ollama. Lire les métadonnées du header.
Produire le bloc `meta` en JSON. Valider sur qwen3.6-coding.
→ Livrable : `extract.py` avec les métadonnées.

### Étape 2 — Statistiques des tenseurs
Itérer tenseur par tenseur (mmap, pas tout en RAM). Calculer mean, std,
norm, min, max, sparsity. Regrouper par couche et composant.
Point d'attention : les tenseurs quantifiés (Q4_K_M, Q8_0) ne sont pas
des floats directs — déquantifier un échantillon ou travailler sur les
blocs bruts selon le type. La lib gguf expose les données via memmap.
→ Livrable : `extract.py` étendu avec `tensor_stats`.

### Étape 3 — Page HTML : fiche de base
Template HTML avec l'en-tête d'identité et les stats architecturales.
Un seul modèle. CSS inline, dark/light theme.
→ Livrable : `template.html` zone 1.

### Étape 4 — Diagramme architectural
SVG généré en JS à partir des métadonnées. Proportionnel, interactif.
→ Livrable : zone 2 intégrée.

### Étape 5 — Spectre des poids
Canvas ou SVG. Grille couche × composant. Hover interactif.
Itérer sur l'encodage visuel.
→ Livrable : zone 3 intégrée.

### Étape 6 — Build et galerie
`build.py` qui assemble data/ dans le template. Navigation entre fiches.
Vue comparative côte à côte optionnelle.
→ Livrable : `build.py` + galerie.

### Étape 7 — Polish et déploiement
Responsive, animations subtiles. Hébergement statique.
→ Livrable : site déployable.

## Points d'attention techniques

- **Mémoire** : Un modèle de 37 Go ne tient pas en RAM déquantifié.
  Itérer tenseur par tenseur via mmap, calculer les stats en streaming.
- **Quantification** : Les poids en Q4_K_M/Q8_0 sont stockés par blocs
  de 32 avec des facteurs d'échelle. La lib gguf les expose via memmap
  mais la déquantification complète peut nécessiter llama-cpp-python.
  À investiguer empiriquement à l'étape 2.
- **Tenseurs MoE** : Les modèles MoE ont des tenseurs d'experts
  (empilés ou individuels). Le spectre doit les représenter — agréger
  les stats sur tous les experts d'une couche, ou montrer les experts
  individuellement. Décision visuelle.
- **Dimensions GGUF** : Les dimensions sont stockées en ordre inversé
  (column-major). Shape [4096, 11008] est stocké [11008, 4096].
