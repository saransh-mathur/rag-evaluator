# AWS S3 Essentials

Amazon S3 is object storage for files of any size. Understanding consistency and access patterns helps control cost and latency.

## Read-after-write consistency

After a successful **PUT** of a new object, S3 provides **read-after-write consistency**: a subsequent GET immediately returns the latest version. Overwrites and deletes are eventually consistent across some edge cases, but new object uploads are strongly consistent for reads.

## Request costs

S3 billing includes storage, requests, and data transfer. Frequently accessed small objects can inflate **request costs** because each GET is charged.

Mitigations:

- Batch objects or use larger payloads when appropriate.
- Put a **CloudFront** distribution in front of public or semi-public buckets to cache at the edge and reduce origin GET volume.
- Use S3 Intelligent-Tiering or lifecycle rules for cold data.

## Security basics

Block public access by default, prefer IAM roles over long-lived keys, and enable server-side encryption (SSE-S3 or SSE-KMS) for sensitive data.
