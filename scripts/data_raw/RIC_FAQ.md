
Taken from: https://apps.embl.de/rbpbase/ -> FAQ

## Why summarizing by gene identifiers and not protein identifiers?

The common identifiers were individually selected for each organism. Because protein identifiers are very unstable, gene identifiers were used to summarize reported RBPs. If any isoform of a protein was reported, the gene was marked as RNA-binding. To account for different genomic locations, unique identifiers were created from the reported gene names. If any of the gene identifiers was reported to be RNA-binding, the unique identifier was marked as RNA-binding. Genes without gene names kept their genomic identifier.

## What Ensembl versions are the identified based on?

The identifiers are based on Ensembl ID version 92

## How was the homology mapping done?

Homology mapping was done with EggNog database[1]. It will be revised soon.

## Why is RBPbase a table browser and not a relational database?

The userbase, use cases and R/Shiny specific properties prefer precompiled tables with optional analytic or display features.
