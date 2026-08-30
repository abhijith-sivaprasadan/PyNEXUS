# Data provenance

The repository currently verifies deterministic synthetic arrays. `config.yaml` contains ENTSO-E and ERA5-oriented settings, but there is no authenticated fetch-and-cache workflow in this version. These settings must be described as configured or planned, not implemented data ingestion.

Any future reference dataset must record its provider, exact variable, geography, period, access time, licence, transformations, missing-data treatment, and checksum. Credentials belong in environment variables and must never be committed.
