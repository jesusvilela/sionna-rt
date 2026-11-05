# Moving–moving frames: a transported polarization projector field

**Checkpoint:** 5 November 2026 · NVIDIA 6G research fork only · CPU/NumPy · no NVlabs modifications, GPU, HF credits or Work.  
**Parent:** `research/munich-cpu-regeneration-boundary-2026-11-05`  
**Source:** Sionna RT PR #82 notebook at fork commit `57435845d3395b2d5e167b0893035e15ffbdaa0d`.

## Verdict and evidence classes

**P (conditional, classical):** gauge-covariant matrix edge differences and transported Bargmann three-projector invariants preserve their mathematical content under independent changes of local input-polarization coordinates. On three nonorthogonal principal eigenlines the invariant can retain orientation/chirality lost by eigenvalue ratios and pairwise overlap magnitudes. This follows from exact matrix conjugation identities; it is established Pancharatnam/Bargmann geometry, not a novel theorem.

**M (bounded):** fourteen local CPU regression tests pass. The two committed Python files match their tested local bytes by Git blob SHA-1. Synthetic projector examples are **not** Munich propagation measurements.

**H:** whether the ordered field of actual Sionna channel principal eigenlines has stable, nontrivial oriented structure that helps interpret cross-receiver behavior under simpler-feature controls.

**R:** the prior assertion that the original fixed realified determinant/null cone uniquely *caused* a symbol-error improvement. Matched optimizer and off-cone controls retired it; this checkpoint does not reanimate that mechanism.

**Blocking gap:** the original notebook did not archive all 16 Munich H matrices. The pinned Sionna CPU-regeneration attempt failed because the sandbox could not download dependencies. No physical-receiver Bargmann phases were produced.

## 1. The typed bundle: do not mix transmit and receive frames

There is one physical complex two-component **transmitter** Jones fiber \(\mathcal V\simeq\mathbb C^2\), and a separate two-component **receiver** fiber \(\mathcal W_v\) at each Munich receiver site \(v\). The channel is the map

\[
H_v:\mathcal V\longrightarrow\mathcal W_v.
\]

The receiver-side choice of frame \(W_v\in U(2)\) changes \(H_v\mapsto W_vH_v\) but **does not change** its transmit-fiber Gram matrix \(G_v=H_v^\dagger H_v\). It therefore leaves all the following objects unchanged.

When \(G_v\) has *distinct* eigenvalues \(\lambda_+>\lambda_-\), it selects the principal input-polarization **projector**

\[
P_v=\frac{G_v-\lambda_- I}{\lambda_+-\lambda_-},
\qquad P_v^\dagger=P_v,\quad P_v^2=P_v,\quad\mathrm{tr}P_v=1.
\]

This formulation avoids choosing a fragile eigenvector phase. Its carrier is the projective line \(\mathbb{CP}^{1}\), not a flat real two-coordinate scatter. At \(\lambda_+=\lambda_-\), **there is no canonical principal eigenline**; code must abstain, not invent one. The executable gap guard is \( (\lambda_+-\lambda_-)/\mathrm{tr}G_v>10^{-7}\).

A **global** transmitter-frame change \(U\in U(2)\) acts as

\[
H_v\mapsto H_vU^\dagger,\quad
G_v\mapsto UG_vU^\dagger,\quad
P_v\mapsto UP_vU^\dagger,\quad C\mapsto UC.
\]

The received codeword differences and their ML error laws do not change.

## 2. Two nested moving frames require explicit transport

It is also mathematically legitimate to re-express each site's common transmit fiber in a *different local coordinate frame* \(U_v\). Those are bookkeeping coordinates, **not sixteen independently configurable transmitters**. We must then carry maps

\[
T_{i\leftarrow j}:\mathcal V_j\to\mathcal V_i,
\qquad
T_{i\leftarrow j}=U_iU_j^\dagger
\]

for the flat reference common transmit trivialization. A second round of site-local frame changes \(V_i\) composes in the order \(U'_i=V_iU_i\); transitions obey \(T'_{i\leftarrow j}=V_iT_{i\leftarrow j}V_j^\dagger\).

The **matrix-valued** first edge difference is

\[
\nabla_{i\leftarrow j}P
=T_{i\leftarrow j}P_jT_{i\leftarrow j}^\dagger-P_i.
\]

It transforms by conjugation in the \(i\)-fiber, rather than collapsing to a scalar norm. Provided the transitions compose \(T_{i\leftarrow k}=T_{i\leftarrow j}T_{j\leftarrow k}\), it satisfies the discrete chain identity

\[
\nabla_{i\leftarrow k}P
=\nabla_{i\leftarrow j}P+
T_{i\leftarrow j}(\nabla_{j\leftarrow k}P)
T_{i\leftarrow j}^\dagger.
\]

When all principal projectors are physically constant and the only apparent motion is a change of coordinates, **the covariant edge difference is zero**. When the projectors actually change in the common physical source basis, it is nonzero. Both controls are tested.

The connection just defined is *flat* and **given by the common source frame**. Its Wilson loop is the identity. It would be mathematically incorrect to call any nonzero diagnostic below a newly measured Maxwell propagation-connection curvature.

## 3. Oriented Bargmann triangle: complex geometry retained

For an oriented triangle \((i,j,k)\) define, with the transports composed exactly as written,

\[
\boxed{
\mathcal B_{ijk}
=\operatorname{tr}
(P_iT_{i\leftarrow j}P_jT_{j\leftarrow k}P_kT_{k\leftarrow i}).
}
\]

Under any sequence of local frame changes, all internal gauges cancel and \(\mathcal B_{ijk}\) remains invariant. Reverse the triangle and the complex value conjugates (for the reciprocal unitary transports used here).

In a **shared transmit basis** (flat \(T=I\)), let the unit-Stokes/Poincaré representatives of the projectors be \(\mathbf n_i,\mathbf n_j,\mathbf n_k\in S^2\). Independently expanding the Pauli matrices yields

\[
\boxed{
\mathcal B_{ijk}
=\tfrac14\!\left[
1+\mathbf n_i\!\cdot\!\mathbf n_j+
\mathbf n_j\!\cdot\!\mathbf n_k+
\mathbf n_k\!\cdot\!\mathbf n_i+
i\,\mathbf n_i\!\cdot(\mathbf n_j\times\mathbf n_k)
\right].
}
\]

The imaginary term is a *three-site oriented* quantity, and its sign reverses under mirror conjugation. The geometric phase is defined as \(\arg\mathcal B\) **only when** \(|\mathcal B|>10^{-10}\). Orthogonal consecutive rays can produce \(\mathcal B=0\), for which assigning phase zero would be a false result.

### Independent exact toy witness

Take the three Jones rays

\[
u_0=(1,0),\quad
u_1=(1,1)/\sqrt2,\quad
u_2=(1,i)/\sqrt2.
\]

Then \(\mathcal B_{012}=(1+i)/4\), so the phase is \(+\pi/4\). Replacing \(u_2\) with its complex conjugate gives \((1-i)/4\), phase \(-\pi/4\), **even though all three pairwise overlap magnitudes and all three individual eigenvalue spectra are identical**.

This does not imply a novel physical chirality effect: complex conjugation is not an arbitrary allowed \(U(2)\) gauge change. It is a *different physical/common-basis projector arrangement* distinguished by an existing geometric invariant.

## 4. Source receiver mesh, without pretending to possess source channels

The source notebook's 16 receivers form **two x columns across eight y levels**. The exact positions define 14 positively oriented triangles (two per y strip) with 29 unique unoriented edges.

The code preserves, when a manifest-verified archive eventually exists:

- all 16 original complex \(2\times2\) transmit Gram matrices, principal projectors and three-component Stokes vectors;
- all **29 matrix-valued edge differences**, including trace-free Hermitian entries, not just their norms;
- all **14 complex triangle invariants**, orientations, magnitudes and phases **or explicit degeneracy/orthogonality abstentions**.

The program requires the prior CPU exporter's `.manifest.json` **and** SHA-256-complete `.npz`, source commit, 16 alive positions, and correct receiver coordinate ordering. Its default mode runs a clearly labeled synthetic three-ray demonstration and marks **per-receiver real evidence unavailable**. It cannot turn one printed notebook channel into 16 measured ones.

### Local-to-global gluing: discrete Pancharatnam–Stokes law [P, conditional]

The 14 oriented triangles form a topological disk: 16 vertices, 29 unoriented edges, 14 faces, hence Euler characteristic 16−29+14=1. The oriented boundary consists of exactly 16 edges. For any triangle whose three overlaps are nonzero, the normalized Bargmann product is the product of its **three normalized, transported edge-overlap amplitudes**.

Multiplying the 14 oriented triangles cancels every internal oriented edge against its reciprocal. Thus

\[
\boxed{
\prod_{(i,j,k)\in F}\frac{\mathcal B_{ijk}}{|\mathcal B_{ijk}|}
=
\prod_{(i,j)\in\partial K}
\frac{\langle u_i,T_{i\leftarrow j}u_j\rangle}
{|\langle u_i,T_{i\leftarrow j}u_j\rangle|}
}
\]

for reciprocal unitary transports (in particular, the flat source-fiber transitions used here). The phases of the arbitrarily selected eigenvector representatives \(u_i\) cancel around the closed boundary. The relation is preserved when each site is put through **two successive moving frames**, provided their transport maps are updated.

It is a **discrete Stokes/gluing identity for complex projector geometry**, not a quantized Chern invariant of the open Munich strip. It is also not a measurement of a Maxwell radiation-field connection. A nonzero phase can arise purely from the trajectory of principal Gram eigenlines even when the bookkeeping transport is flat.

The new function returns both the complete oriented boundary and both complex products, together with their difference, refusing to assign phases if an edge or face overlap vanishes. Its local CPU randomized regression verifies a residual below \(10^{-12}\) in both original and independently reframed coordinates.

## 5. Test and adversarial controls

Reproduce from the fork root, with NumPy only:

```bash
python -m unittest discover -s research/tests -p 'test_moving_frame.py' -v
python research/experiments/moving_frame.py
# only if a verified real archive exists:
python research/experiments/moving_frame.py --channels-npz path/to/munich_channels.npz --out receiver_moving_frames.json
```

Sixteen tests check:

1. exact \((1\pm i)/4\) values and reversed triangle;
2. mirror with identical pairwise overlap magnitudes and opposite phase;
3. a **separate Stokes/Pauli reference formula** across 300 random complex triples;
4. 100 global \(U(2)\) gauge changes;
5. 50 *independent site-local* \(U(2)\) gauges versus a deliberately wrong untransported product;
6. two successive moving frames with the correct gauge-composition order;
7. 70 independent principal-eigenprojector reconstructions, degeneracy and malformed Gram rejection;
8. undefined phase handling at zero overlap/eigenvalue degeneracy;
9. matrix-valued edge composition, physical-vs-coordinate motion and nonunitary transport rejection;
10. the orientation of all 14 source triangles;
11. the structure and Hermiticity of all 29 stored matrix-valued edge changes;
12. invariance under independent receive-side left unitaries;
13. rejection of malformed/unnormalized channel arrays;
14. prohibition on silently loading an unverified/missing Munich archive.

Additional tests (15–16) certify the discrete 14-face-to-16-boundary complex gluing identity under independent local gauges, and enforce an explicit abstention for zero-overlap paths. The tests are local, AI-authored and correlated with the implementation; exact algebra and adversarial metamorphic witnesses are more substantial than a pass count alone.

**Local Git-blob SHA-1 of tested source:** `3887ace7c0c1ac43e0e94a50a65253661f2444ff`.  
**Local Git-blob SHA-1 of tested tests:** `ce035ec322133e372bbdc3f9888d850b7fa656e4`.  
Both have been compared against fetched GitHub file blobs.

## 6. Boundary of scientific meaning / next smallest discriminating step

This invariant is a *spatial three-site statistic of principal channel eigenpolarizations*. It is not, without additional measured physics, an optical phase accumulated by propagating radiation, a nontrivial transport of Maxwell field paths, nor an improvement in the one-shot coherent SER predictor: **the complete per-site Gram already fixes that receiver's SER under the existing white-noise model**.

The extra mathematical content is relative to **reduced summaries** (eigenvalues, individual polarization strengths and pairwise absolute overlaps). Full ordered Grams already contain the entire invariant.

A future matched test should hold out spatial receiver strips and compare (a) reduced pairwise Stokes/Gram features, (b) these triad orientation features, and (c) the **full ordered Gram field** on a task that actually uses *spatial relations*, such as predicting the next receiver's principal eigenpolarization or joint outage patterns. If the richer feature adds no out-of-sample value over equally costed controls, retire its proposed engineering usefulness. The 16-source-channel blocker must be resolved before any such empirical promotion.

**Source/prior-art context:** Pancharatnam connections and half-solid-angle phases are established in optical polarization; see [van Dijk, Schouten and Visser, JOSA A 27 (2010)](https://doi.org/10.1364/JOSAA.27.001972) and [Kestner and Kostinski, Physical Review A 109 (2024)](https://doi.org/10.1103/PhysRevA.109.023517). This application does not assert independent novelty.

**Stop and resource gate:** no NVlabs comment/PR, no touch to PR #82, no GPU, no HF credits and no Work use. CPU tests and repository commits only.
