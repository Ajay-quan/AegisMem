# ADR 0001: Single-Node AWS Free Tier Demonstration

## Status

Accepted

## Context

stateful.ai needs to prove cloud deployability without creating cost. Managed services such as RDS, OpenSearch, EFS, NAT Gateway, load balancers, Route 53 hosted zones, and API Gateway can improve production architecture but violate the zero-cost constraint.

## Decision

Deploy the demo as one Docker container on one EC2 Free Tier instance with an 8 GB gp3 EBS-backed local data directory. Keep internal service boundaries in code, but do not split infrastructure into multiple paid services for the portfolio demo.

## Consequences

- The demo is inexpensive and easy to tear down.
- FAISS, graph, and JSON persistence stay on local EBS.
- The deployment is not horizontally distributed.
- Resume wording should say "microservices-style architecture; single-node Free Tier demo deployment."
