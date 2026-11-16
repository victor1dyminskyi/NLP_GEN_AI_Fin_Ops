Data dictionary (what the important columns mean). Used in every LLM prompt that writes SQL.

billing — one row = one resource on one DAY (FOCUS format)
- ChargePeriodStart: the DAY of usage. Use this for any question about dates or days.
- BillingPeriodStart / BillingPeriodEnd: first day of the invoice MONTH. Do NOT use for daily questions.
- EffectiveCost: cost in USD of that resource on that day. "cost" always means EffectiveCost.
- PricingQuantity: amount used that day (hours for servers, GB for storage/traffic).
- ContractedUnitPrice: price per unit (per hour / per GB) after discounts. ListUnitPrice = public price.
- PricingCategory: 'Committed' = covered by a Savings Plan, 'Standard' = on-demand price.
- CommitmentDiscountId: which Savings Plan covered the row (NULL if none).
- ResourceName: human-readable name, e.g. 'web-server-1'. Use it when a question names a resource.
- ResourceId: AWS id, e.g. 'i-0abc…'. SubAccountName: AWS account name. ServiceName: AWS service.
- Tags: JSON text like {"team": ..., "application": ..., "environment": ...}, or NULL when untagged.

usage_metrics — one row = one server on one day it was running
- date: the day. cpu_pct / gpu_pct: average % busy that day (gpu_pct NULL for non-GPU servers).
- instances_avg: average number of instances running.

change_log — CloudTrail-style events
- event_time: timestamp. actor: username. actor_type: 'user' or 'automation'.
- event_name uses AWS names: RunInstances (server created), TerminateInstances (server deleted),
  StopInstances / StartInstances (scheduler), CreateVolume, AttachVolume, DeleteVolume, CreateBucket,
  CreateDBInstance, CreateSnapshot, CreateNatGateway, CreateLoadBalancer, ChangeApproved (approved change).
- ticket: change ticket text (e.g. 'CHG-1234: …') or NULL. details: extra text (e.g. 'attached to <server>').

owners — one row per team
- team, owner, username, email, slack.
- accounts: COMMA-SEPARATED list of SubAccountName values owned by the team.

commitments — Savings Plans
- commitment_id, type, discount (0.25 = 25%), start, "end" (quote it: end is an SQL keyword).
- covers: COMMA-SEPARATED list of ResourceName values. managed_by: team responsible for renewing.
