# Publishing workflow

The user requested that completed project updates be pushed to the matching GitHub repository after validation:

- Rebuilt/new SmartRMA project (this directory): `HappyDeswal/SmartRMA_Publish`.
- Original/old SmartRMA project (the separate original checkout): `HappyDeswal/SmartRMA_Final`.

Verify the target remote before publishing. When this directory is nested inside the original checkout, publish its application files at the root of SmartRMA_Publish; do not commit them into SmartRMA_Final by mistake. Preserve remote commits and push without force.

Keep private runtime state, credentials, customer histories, environments, downloaded datasets, and generated weights excluded according to `.gitignore`. Publish the source, training notebook/scripts, documentation, and non-private evaluation evidence. Report any updates that could not be pushed.
