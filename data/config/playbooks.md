FinOps playbook — how to classify an alert and what to do. Used in the agent's instructions.
General FinOps knowledge (patterns and remedies), not answers for any specific dataset.

HOW TO RECOGNISE THE RESOURCE KIND (from resource_info price_line / cost_history unit)
- server: price_line contains "Instance Hour" (GPU servers: g5, p3, p4, g4dn instance types).
- disk: price_line contains "GB-month of … storage" or "snapshot".
- data transfer / traffic: unit is GB per day and price_line mentions "data transfer", "Data Processed"
  or "log data ingested"; its cost moves with traffic.

TYPES (choose exactly one)
- idle_gpu: a GPU server (instance type g5/p3/p4/g4dn) is billed every day but gpu_pct stays near 0
  for days (e.g. after a training run). Ask: is it still billed? is it busy? is there an approved ticket?
- transfer_spike: data-transfer / traffic cost (GB) jumps sharply for some days, price per GB unchanged,
  no approved ticket.
- transfer_drift: data-transfer / traffic cost (GB) rises gradually over 1–3 weeks and stays high,
  no approved ticket.
- untagged_resource: a new resource with real cost has NO tags (Tags is NULL), so nobody is accountable.
  Find who created it in the change log.
- sp_expired: price per unit rises clearly while usage (hours) stays about the same; PricingCategory changes
  from 'Committed' to 'Standard'; the Savings Plan's end date is at the start of the change.
- orphaned_disk: a disk (EBS volume) is still billed although the server it was attached to was terminated.
- planned_change: the cost increase is covered by an APPROVED change ticket (ticket text, ChangeApproved
  event, or ticket on the RunInstances event) and matches what the ticket says. Not a problem.
- no_issue: the cost is legitimate and explained (e.g. new tagged resource with a known owner and normal
  usage) but has no ticket and no sign of waste. Not a problem.
- other: none of the above fits (a human will review it — prefer a specific type when the evidence fits).

ACTIONS (choose exactly one; the approval tier is set automatically by code)
- stop_instance: for idle_gpu.
- notify_owner_open_ticket: for transfer_spike and transfer_drift.
- tag_and_notify_owner: for untagged_resource.
- recommend_sp_renewal: for sp_expired.
- snapshot_then_delete_volume: for orphaned_disk.
- no_action: for planned_change and no_issue.

OWNER
- The accountable cost owner is the TEAM that owns the AWS account the resource runs in
  (owners.accounts). For untagged resources use the account's team as well.
- For sp_expired the owner is the team in commitments.managed_by.

CONFIDENCE (0–1)
- 0.9+: evidence clearly matches one type. 0.6–0.9: likely. Below 0.6: unsure → a human will review.
