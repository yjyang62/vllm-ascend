# Experimental HCCP service lease without an anchor group

This extra-cleanup path keeps the HCCP service alive across destruction of all
business HCCL groups. It does not create an anchor or preserve a business group.
Business communicators and ACL graphs are recreated on wake-up.

It uses internal CANN runtime symbols through a process-local preload shim.
Only the tested A3 / CANN 9.1.0 / driver 26.0.rc1 setup is covered. This is not
a supported public CANN API or a production compatibility guarantee.

Build on the target ARM64 host, using the matching CANN headers:

```bash
gcc -shared -fPIC -O2 \
  -I /usr/local/Ascend/cann-9.1.0/aarch64-linux/include \
  -I /usr/local/Ascend/cann-9.1.0/aarch64-linux/pkg_inc \
  csrc/sleep_hccp_lease/net_lease.c -o /absolute/path/libnet_lease.so -ldl -lpthread
```

Load the shim into the serving process and its worker children with
`LD_PRELOAD=/absolute/path/libnet_lease.so`, and enable:

```json
{"rl_config":{"enabled":true,"sleep_mode_extra_cleanup":true}}
```

`sleep_mode_extra_cleanup` is off by default. When it is enabled, missing native
symbols or an invalid lease transition raise an error; the implementation never
silently creates an anchor.
The HCCP service remains allocated while asleep, but its business communicators
are destroyed. The shim defers runtime Close while held and reuses the service
on a matching Open. The final unheld Close reaches the real runtime.

Validated: three level-1 sleep/wake cycles, including weights-then-KV staged
wake, on the three-layer DeepSeek reproducer. Model buffer and four fused slot
metadata restorations are included for that reproducer. This is not a complete
persistent-buffer audit for other models. Level-2 updates, other hardware and
toolkit versions, multi-node operation, and fault recovery are not validated.

The native service lease avoids the trigger of the known topic PID overwrite.
It does not repair device-side mapping logic and does not preserve graphs.
