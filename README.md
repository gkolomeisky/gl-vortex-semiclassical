# GL Vortex: Exact Single-Scale Outer Solution in the Extreme Type-II Limit

Companion code for the paper:

> G. Kolomeisky, "Exact Single-Scale Outer Solution of the Abrikosov Vortex
> in the Extreme Type-II Limit", *Physical Review Letters* (submitted, 2026).

## Physics summary

This code demonstrates that in the extreme type-II limit (κ → ∞), the
Abrikosov vortex solution outside a vanishing core of radius ξ = 1/κ is
given exactly by:

    v(r) = K₁(r)/κ
    b(r) = K₀(r)/κ
    R²(r) = 1 − K₁²(r)/κ²

where r is measured in units of the London penetration depth λ, v is the
superfluid velocity, b is the magnetic induction, and R is the
order parameter amplitude. This constitutes a single-scale outer solution —
both the superfluid density deficit and the magnetic field decay on the
scale λ, contradicting the standard two-scale (ξ and λ) picture.

## Units and conventions

- Length unit: λ (London penetration depth), so λ = 1
- κ = λ/ξ is the Ginzburg-Landau parameter
- v denotes the gauge-invariant superfluid velocity (not the vector potential)
- The GL equations are solved as a boundary value problem in the
  logarithmic coordinate s = ln(r), with W(s) = r·v(r)

## Files

| File | Description |
|------|-------------|
| `figure1_deGennes.py` | de Gennes-style log-r, linear-y plot of R² and b/b(0) for κ=20 |
| `figure2_convergence.py` | Two-panel convergence figure: κb → K₀ and κ²(1−R²) → K₁² for κ=5,10,20,40 |
| `requirements.txt` | Python dependencies |

## Usage

Install dependencies:

    pip install -r requirements.txt

Run each script from the command line:

    python figure1_deGennes.py
    python figure2_convergence.py

Each script saves a publication-quality PDF in the current directory.

## Dependencies

- Python 3.8+
- numpy, scipy, matplotlib (see requirements.txt)

## Acknowledgment

Numerical coding was produced with the assistance of Claude (Anthropic).

## License

MIT License — free to use, modify, and distribute with attribution.
