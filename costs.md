# Hosting costs (rough estimates)

Estimates as of October 2026. Most numbers are **assumed**, not measured on a server. Two are measured on
a laptop (i5-12450H, 12 threads):

- one recording of three verses transcribed at about **26× real time**;
- under load, the server handled about **42 seconds of audio per second**.

## RAM and compute

| Component | Estimated RAM |
|---|---|
| fp32 model in ONNX Runtime (480 MB on disk) | ~0.7–1.0 GB |
| Python, FastAPI, Silero VAD, ffmpeg runs, request buffers | ~0.3–0.5 GB |
| **Total** | **~1–1.5 GB; plan on 4 GB for headroom** |

- **No GPU needed.** The CPU is the limit.
- **Capacity:** each person speaking sends 1 second of audio per second.

  | Server size | Assumed speed | People speaking at once |
  |---|---|---|
  | 2 vCPU | ~8–10× real time | ~5–8 |
  | 4 vCPU | ~15–20× | ~12–15 |
  | 8 vCPU (dedicated) | ~35–40× | ~30+ |

  Most visitors are silent most of the time, so a 2 vCPU server should handle dozens of open tabs.
- **Disk:** the model, Python environment and OS need about 5 GB; a 40 GB disk is plenty.
- **Bandwidth:** about 32 KB/s upload per active speaker, which is negligible.

## Monthly cost, running 24×7

| Option | Specs | Approx. cost/month | Good for |
|---|---|---|---|
| Oracle Cloud Always Free (Arm) | Reportedly cut to 2 OCPU / 12 GB in June 2026 | **Free** | Demo or personal use, if you can get capacity |
| Hetzner cost-optimised (CX23-class) | 2 vCPU / 4 GB | **~€4–6** | Small, low traffic |
| Hetzner regular or Arm (CPX22 / CAX21-class) | 2–4 vCPU / 4–8 GB | **~€7–10** | **Recommended start** |
| DigitalOcean Basic | 2 vCPU / 4 GB | **$24** | Simpler setup, pricier |
| DigitalOcean Basic | 4 vCPU / 8 GB | **$48** | ~10–15 people speaking at once |
| Hetzner dedicated vCPU | 4–8 dedicated vCPU | **~€45–90** | Heavy, steady traffic |
| AWS / GCP equivalent | 2–4 vCPU / 4–8 GB | **~$30–70** | Only if you're already on that cloud |

Avoid burstable instances like AWS t3/t4g. Transcription maxes out the CPU, and you'd run out of CPU
credits under steady load.

Hetzner raised prices in April 2026 and again for new orders from 15 June 2026. Treat these figures as
roughly right and check the live pricing page before ordering.

**Small extras:**

- Domain name: about $10–15 a year.
- HTTPS: free with Let's Encrypt (Caddy sets it up automatically). It's required, because browsers only
  allow the mic on HTTPS.
- Server backups: about 20% on top of the server price.

## Storing the transcriptions

| What you store | Size | 100 hours of speech per month |
|---|---|---|
| **Text only** (Devanagari, about 15 characters per second) | ~160 KB per hour of speech | ~16 MB per month, effectively free in SQLite or Postgres on the same server |
| Audio as WAV (16 kHz) | ~115 MB per hour | ~11.5 GB per month |
| Audio as Opus (24 kbps) | ~11 MB per hour | ~1.1 GB per month |

- **Text:** costs nothing extra; even years of it stay in the low GB.
- **Audio:** store it as Opus in object storage, which costs about $0.015/GB-month on Cloudflare R2 (with
  free downloads), about $0.023 on S3, or around €5/TB on Hetzner. At 100 hours a month, that's under
  $1/month for years.
- **Why keep audio:** with the correct transcripts, it's the data you'd need to measure accuracy or
  fine-tune later.
- **Privacy:** storing people's voices needs a consent notice and a deletion option in the UI.

## Bottom line

- **Personal or demo:** Oracle free tier, or a ~€5 Hetzner server. Total about **€0–6/month**.
- **Real users (up to ~15 speaking at once):** a 4 vCPU / 8 GB server at about **€8–10/month** on Hetzner
  or **$48** on DigitalOcean. Storing text and audio adds about **$1/month**.

To firm these up, measure RAM and 2- and 4-core throughput on the target machine before choosing a plan.

## Sources

- [Hetzner Cloud](https://www.hetzner.com/cloud)
- [Hetzner "What's New" (15 June 2026 price change)](https://docs.hetzner.cloud/whats-new)
- [Hetzner pricing 2026 summary](https://agentdeals.dev/hetzner-pricing-2026)
- [Igor'sLAB on the April 2026 Hetzner increase](https://www.igorslab.de/en/hetzner-to-significantly-increase-prices-for-cloud-and-dedicated-servers-from-april-2026/)
- [Bitdoze: Hetzner cost-optimised plans](https://www.bitdoze.com/hetzner-cloud-cost-optimized-plans)
- [Infratally: DigitalOcean pricing 2026](https://infratally.com/articles/digitalocean-droplet-pricing-guide-2026/)
- [Temperstack: DigitalOcean plans](https://www.temperstack.com/plans/digitalocean/)
- [InfoQ: Oracle free tier limits (July 2026)](https://infoq.com/news/2026/07/oracle-cloud-free-tier-limits/)
