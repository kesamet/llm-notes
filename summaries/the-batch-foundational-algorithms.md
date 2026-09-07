# The Batch: Foundational Algorithms, Where They Came From, Where They're Going -- Wiki

> Based on Andrew Ng's article (April 2022)
> Source: https://info.deeplearning.ai/the-batch-special-issue-foundational-algorithms-where-they-came-from-where-theyre-going-2

## Table of Contents

- [Overview](#overview)
- [Linear Regression](#linear-regression)
  - [History](#history)
  - [Mechanism](#mechanism)
  - [Variants](#variants)
- [Logistic Regression](#logistic-regression)
  - [History](#history-1)
  - [Mechanism](#mechanism-1)
  - [Extensions](#extensions)
- [Gradient Descent](#gradient-descent)
  - [History](#history-2)
  - [Mechanism](#mechanism-2)
  - [Challenges and Variants](#challenges-and-variants)
- [Neural Networks](#neural-networks)
  - [History](#history-3)
  - [Architecture](#architecture)
  - [Training](#training)
  - [Limitations](#limitations)
- [Decision Trees](#decision-trees)
  - [History](#history-4)
  - [Mechanism](#mechanism-3)
  - [Ensemble Methods](#ensemble-methods)
- [K-Means Clustering](#k-means-clustering)
  - [History](#history-5)
  - [Mechanism](#mechanism-4)
  - [Variants](#variants-1)
- [Key Takeaways](#key-takeaways)
- [References](#references)

---

## Overview

This special issue of *The Batch* surveys six foundational machine learning algorithms that form the bedrock of modern AI systems: **linear regression**, **logistic regression**, **gradient descent**, **neural networks**, **decision trees**, and **k-means clustering**. Understanding these algorithms—their origins, mechanics, and evolution—is essential for effective ML engineering. The article traces each algorithm from its mathematical inception to its role in contemporary deep learning.

---

## Linear Regression

### History

| Year | Contributor | Contribution |
|------|-------------|--------------|
| 1805 | Adrien-Marie Legendre | Published method of fitting a line to points (comet prediction) |
| 1795 (claimed) | Carl Friedrich Gauss | Asserted prior use; deemed "too trivial to write about" |
| 1922 | Ronald Fisher & Karl Pearson | Integrated into general statistical framework of correlation and distribution |
| Late 20th century | — | Computers enabled large-scale application |

**Priority dispute**: Legendre published first (1805); Gauss claimed earlier use (1795). The matter remains unresolved.

### Mechanism

Linear regression models a linear relationship between outcome $y$ and input $x$:

$$y = w \cdot x + b$$

- **$w$ (slope/weight)**: How steeply $y$ changes with $x$
- **$b$ (bias/intercept)**: $y$ at $x = 0$

**Training**: Given $(x, y)$ pairs, predict $\hat{y}$, compute squared error $(\hat{y} - y)^2$, minimize via **ordinary least squares** (OLS) to optimize $w$ and $b$.

**Multivariate extension**: Adding features (e.g., car drag) extends the line to a hyperplane: $y = w_1 x_1 + w_2 x_2 + \dots + b$.

### Variants

| Variant | Regularization | Effect | Use Case |
|---------|----------------|--------|----------|
| **Ridge (L2)** | $\lambda \sum w_i^2$ | Shrinks coefficients evenly; discourages reliance on any single feature | Good default; correlated features |
| **Lasso (L1)** | $\lambda \sum \|w_i\|$ | Drives coefficients to zero; performs feature selection | Sparse data; interpretability |
| **Elastic Net** | Combines L1 + L2 | Balances selection and grouping | High-dimensional, correlated features |

**Deep learning connection**: The standard neuron computes $w^T x + b$ followed by a nonlinear activation—linear regression is the core building block.

---

## Logistic Regression

### History

| Year | Contributor | Contribution |
|------|-------------|--------------|
| 1830s | P.F. Verhulst | Invented logistic function for population dynamics (S-curve) |
| Early 20th c. | E.B. Wilson & Jane Worcester | Devised logistic regression for lethal dose estimation |
| Late 1960s | David Cox & Henri Theil | Extended to multinomial outcomes (independently) |

### Mechanism

Logistic regression fits the **logistic function** (sigmoid) to predict probability of a binary outcome:

$$P(y=1|x) = \frac{1}{1 + e^{-(w^T x + b)}}$$

- **Center (horizontal shift)**: Dose required for 50% probability
- **Slope (steepness)**: Certainty of transition; steep = sharp threshold, gentle = gradual

**Classification**: Apply threshold (default 0.5) to probability output → binary decision.

### Extensions

| Extension | Description |
|-----------|-------------|
| **Multinomial logistic regression** | >2 unordered outcomes (Cox, Theil) |
| **Ordered logistic regression** | >2 ordered outcomes |
| **Regularized logistic regression** | L1/L2/Elastic Net for sparse/high-dimensional data |

**Applications**: Medicine (mortality risk), political science (election prediction), economics (business forecasts), neural network neurons (sigmoid activation).

---

## Gradient Descent

### History

| Year | Contributor | Context |
|------|-------------|---------|
| 1847 | Augustin-Louis Cauchy | Approximating stellar orbits |
| ~1907 | Jacques Hadamard | Deformations of thin flexible objects |
| Modern | — | Minimizing loss functions in ML |

### Mechanism

Iterative optimization to minimize a loss function $L(\theta)$:

1. **Position** = current parameters $\theta$
2. **Altitude** = loss $L(\theta)$
3. **Gradient** $\nabla L(\theta)$ = direction of steepest *ascent*
4. **Update**: $\theta \leftarrow \theta - \alpha \nabla L(\theta)$
   - $\alpha$ = **learning rate** (step size)

**Trade-off**: Small $\alpha$ → slow convergence; large $\alpha$ → oscillation/divergence.

### Challenges and Variants

| Challenge | Description | Solution |
|-----------|-------------|----------|
| **Local minima** | Nonconvex landscapes trap optimization | Momentum (accelerates past small barriers) |
| **Saddle points** | Flat regions with zero gradient | Adaptive learning rates (Adam, RMSprop) |
| **Plateaus** | Vanishing gradients | Learning rate schedules, batch normalization |
| **Ill-conditioning** | Ravine-shaped loss landscapes | Preconditioning, second-order methods |

**Key insight**: In deep learning, local and global minima often yield similar performance. Gradient descent remains the universal optimizer—exact solutions (e.g., OLS for linear regression) exist but gradient descent often converges faster at scale.

---

## Neural Networks

### History

| Year | Contributor | Milestone |
|------|-------------|-----------|
| 1873 | — | Insight: brain learns via neuron interactions |
| 1943 | Warren McCulloch & Walter Pitts | Mathematical model of biological neurons |
| 1958 | Frank Rosenblatt | Perceptron (single-layer, punch-card implementation) |
| 1965 | Alexey Ivakhnenko & Valentin Lapa | Multi-layer networks (overcame linear separability limit) |
| 1985–86 | Yann LeCun, David Parker, David Rumelhart et al. | Backpropagation for efficient training |
| 1970s–80s | Seppo Linnainmaa, Paul Werbos | Earlier automatic differentiation foundations |
| 2000s | Kumar Chellapilla, Dave Steinkraus, Rajat Raina, Andrew Ng | GPU acceleration enabling large-scale training |

### Architecture

A neural network is a **trainable function** composed of simple neuron functions:

**Neuron computation**:
$$z = w^T x + b \quad \text{(linear regression)}$$
$$a = \sigma(z) \quad \text{(activation: ReLU, sigmoid, tanh, etc.)}$$

- **Weights $w$**: Adjustable parameters determining the function
- **Activation $\sigma$**: Nonlinearity enabling universal approximation
- **Layers**: Stack neurons; output of layer $l$ = input to layer $l+1$

### Training

1. **Forward pass**: Compute output for input batch
2. **Loss**: Compare output to targets (e.g., cross-entropy, MSE)
3. **Backpropagation**: Compute $\nabla_\theta L$ via chain rule
4. **Update**: $\theta \leftarrow \theta - \alpha \nabla_\theta L$ (gradient descent)
5. **Repeat** until convergence

### Limitations

- **Common sense & logical reasoning**: GPT-3 fails "what comes before a million?" (answers 999,999)
- **Data hunger**: Requires massive labeled datasets
- **Interpretability**: Black-box decisions
- **Brittleness**: Adversarial examples, distribution shift

---

## Decision Trees

### History

| Era | Contributor | Contribution |
|-----|-------------|--------------|
| 3rd century | Porphyry | Logical classification tree (Aristotle's categories) |
| 1963 | John Sonquist & James Morgan | First computerized decision trees (survey analysis) |
| 1986 | John Ross Quinlan | ID3: nonbinary outcomes, information gain |
| 2001 | Leo Breiman & Adele Cutler | Random Forest (ensemble of trees) |
| 2008 | — | C4.5 named Top 10 Algorithm in Data Mining (IEEE) |

### Mechanism

**Structure**: Root node → decision nodes → leaf nodes (predictions)

**Training (recursive partitioning)**:
1. At each node, evaluate all features/thresholds
2. Choose split maximizing **purity** (e.g., Gini impurity, entropy reduction)
3. Recurse on child nodes until purity plateaus or max depth

**Inference**: Traverse tree from root to leaf; predict leaf's majority class (classification) or mean value (regression).

### Ensemble Methods

| Method | Mechanism | Benefit |
|--------|-----------|---------|
| **Random Forest** | Bootstrap samples + random feature subsets per tree; majority vote | Reduces overfitting, variance |
| **XGBoost / Gradient Boosting** | Sequential trees correcting residuals; weighted vote | State-of-the-art tabular performance |

**Trade-offs**: Single trees overfit and are unstable (small data change → different tree). Ensembles solve both.

---

## K-Means Clustering

### History

| Year | Contributor | Context |
|------|-------------|---------|
| 1957 | Stuart Lloyd | Bell Labs / Manhattan Project; digital signal quantization (unpublished until 1982) |
| 1965 | Edward Forgy | Similar method (Lloyd-Forgy algorithm) |

### Mechanism

**Input**: Data points $\{x_i\}$, number of clusters $k$

**Algorithm (Lloyd's)**:
1. **Initialize**: Randomly select $k$ centroids
2. **Assign**: Each point → nearest centroid (Euclidean distance)
3. **Update**: Centroid = mean of assigned points
4. **Repeat** steps 2–3 until centroids stabilize

**Inference**: New point → nearest centroid

**Distance metric**: Any valid metric (cosine, Manhattan, custom kernels)—not limited to spatial distance.

### Variants

| Variant | Centroid Definition | Property |
|---------|---------------------|----------|
| **K-medoids** | Actual data point minimizing intra-cluster distance | Interpretable centroids; robust to outliers |
| **Fuzzy C-Means** | Soft assignment: degree of membership $\in [0,1]$ per cluster | Handles overlapping clusters |

**Acceleration**: KD-trees (2002) partition high-dimensional space for $O(\log n)$ nearest-centroid search.

**Advantage**: Unsupervised—no labels required.

---

## Key Takeaways

1. **Foundational algorithms persist**: Linear/logistic regression, gradient descent, and decision trees underlie modern deep learning—neurons *are* regularized linear/logistic units; backprop *is* gradient descent.

2. **History informs practice**: Priority disputes (Legendre vs. Gauss), independent rediscovery (Cauchy/Hadamard, Cox/Theil), and delayed recognition (Lloyd) are normal in ML. Revisiting fundamentals prevents costly misjudgments (Ng's boosted trees anecdote).

3. **Regularization is universal**: L1/L2/Elastic Net apply across linear regression, logistic regression, and neural networks (weight decay) for the same reasons: prevent overfitting, handle collinearity, enable feature selection.

4. **Ensembles beat single models**: Random Forest and XGBoost transform high-variance decision trees into robust predictors—same principle as model averaging in deep learning.

5. **Unsupervised learning scales**: K-means requires no labels and accelerates via spatial indexing (KD-trees), making it practical for massive datasets where labeling is infeasible.

6. **Hardware drives adoption**: GPUs (2000s) enabled neural networks; KD-trees (2002) accelerated k-means; computational advances unlock theoretical algorithms.

7. **Neural networks lack systematic reasoning**: Despite superhuman performance on pattern recognition (Go, medical imaging), they fail basic logic—a known gap since the perceptron era.

---

## References

- Legendre, A.M. (1805). *Nouvelles méthodes pour la détermination des orbites des comètes*
- Gauss, C.F. (1809). *Theoria motus corporum coelestium*
- Fisher, R. & Pearson, K. (1922). On the mathematical foundations of theoretical statistics
- Verhulst, P.F. (1838). Notice sur la loi que la population poursuit dans son accroissement
- Wilson, E.B. & Worcester, J. (1943). The law of survival
- Cox, D.R. (1966). Some procedures associated with the logistic qualitative response curve
- Theil, H. (1969). A multinomial extension of the linear logit model
- Cauchy, A.L. (1847). Méthode générale pour la résolution des systèmes d'équations simultanées
- McCulloch, W. & Pitts, W. (1943). A logical calculus of the ideas immanent in nervous activity
- Rosenblatt, F. (1958). The perceptron: a probabilistic model for information storage and organization
- Ivakhnenko, A. & Lapa, V. (1965). Cybernetic predicting devices
- Rumelhart, D., Hinton, G. & Williams, R. (1986). Learning representations by back-propagating errors
- Linnainmaa, S. (1970). The representation of the cumulative rounding error of an algorithm as a Taylor expansion of the local rounding errors
- Werbos, P. (1974). Beyond regression: new tools for prediction and analysis in the behavioral sciences
- Quinlan, J.R. (1986). Induction of decision trees
- Breiman, L. (2001). Random forests
- Lloyd, S. (1982). Least squares quantization in PCM
- Forgy, E. (1965). Cluster analysis of multivariate data: efficiency vs interpretability
- Ng, A. (2022). The Batch: Special Issue—Foundational Algorithms. deeplearning.ai