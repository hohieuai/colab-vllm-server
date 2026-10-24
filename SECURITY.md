# Security policy

## Supported versions

Security fixes go into the latest release only. Upgrade to it before you report a problem.

| Version | Supported |
|---|---|
| 0.1.x | Yes |

## Report a vulnerability

Report vulnerabilities privately. Don't open a public issue, discussion, or pull request, because
that discloses the problem before a fix exists.

1. Open the [private reporting form](https://github.com/hohieuai/colab-vllm-server/security/advisories/new).
   You can also reach it from the repository's **Security** tab, under **Report a vulnerability**.
2. Describe the problem and its impact. Say what an attacker can do, for example reach the server
   from outside your computer.
3. List the steps to reproduce it. Include your operating system, the preset, and the vLLM version
   from the server log.
4. Name the versions you know are affected, and any mitigation or fix you have found.

Only the maintainer and you can see the report. If you can't use GitHub, open a public issue that
asks for a private contact and contains no details of the problem.

## What happens after you report

1. The maintainer acknowledges your report within 7 days.
2. Within 14 days, the maintainer confirms the vulnerability or explains why it isn't one.
3. If it's confirmed, the maintainer develops the fix privately, releases it, and publishes a
   GitHub Security Advisory.
4. The advisory credits you by name, unless you prefer to stay anonymous.

## Disclosure timeline

This project follows coordinated disclosure with a 90-day limit, as the
[OpenSSF vulnerability disclosure guide](https://github.com/ossf/oss-vulnerability-guide)
recommends. The fix and the advisory are published within 90 days of your report, and earlier when
the fix is ready.

## Scope

These reports are in scope:

- The `serve` command, the resolver, and the files they write.
- `remote/launch.py` and how it runs the server on the VM.
- The SSH tunnel setup and how the API key and `HF_TOKEN` are stored and passed.

These reports are out of scope:

- Vulnerabilities in vLLM, the `colab` CLI, Colab, or vLLM Recipes. Report those to their own
  projects.
- The default API key `vllm`. The server listens only on the VM's loopback interface and is
  reached through your SSH tunnel. Set `VLLM_API_KEY` if your setup exposes the local port.
- Attacks that need administrator or physical access to your computer.
