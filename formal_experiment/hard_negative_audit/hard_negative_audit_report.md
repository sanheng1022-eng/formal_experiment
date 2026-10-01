# LLM-assisted Hard-negative Label Audit — Completed Summary

LLM-reviewed audit samples: **494**. The frozen sample definition and source hashes are in `audit_manifest.json`.

| Label | Count | Proportion |
|---|---:|---:|
| clearly unaffected | 418 | 84.6% |
| plausibly affected / possible missing secondary gold | 35 | 7.1% |
| uncertain | 41 | 8.3% |

## Representative labelled records

### clearly unaffected

- `009b7c99-b777-465a-bc3a-77f2d88c1318` → `14ed299f-bf3b-45b3-bf7f-8265adebbd87`; sources: critical_harmful_transition; reviewer note: Wearing a walking boot does not change the user's unease about being above the second floor or their avoidance of such places.. 
- `009b7c99-b777-465a-bc3a-77f2d88c1318` → `26910318-c4ce-431f-8b6f-8eaad3ffd8b8`; sources: critical_harmful_transition; reviewer note: Wearing a walking boot does not change whether the user's location typically has snow on the ground from December through March.. 
- `009b7c99-b777-465a-bc3a-77f2d88c1318` → `309111b3-4d14-46e4-bffd-3841e033a9bf`; sources: critical_harmful_transition; reviewer note: Wearing a walking boot does not change whether the user's place is rarely cold for this time of year.. 

### plausibly affected / possible missing secondary gold

- `009b7c99-b777-465a-bc3a-77f2d88c1318` → `eda4a6c5-47eb-489c-ad47-35a533f23acb`; sources: critical_harmful_transition; reviewer note: Wearing a walking boot could prevent the user from maintaining a current every-other-day jogging routine, making that routine's current status worth revisiting.. 
- `309111b3-4d14-46e4-bffd-3841e033a9bf` → `68f30210-15b0-4dcb-b381-a51523b6175a`; sources: calibration_global_top30 | multipic_global_top30; reviewer note: Moving to a new place directly contradicts the historical statement that the user did not plan to move from their current address.. 
- `309111b3-4d14-46e4-bffd-3841e033a9bf` → `dafb876f-f85e-4d8b-8522-71a4bbd5a492`; sources: calibration_global_top30 | multipic_global_top30; reviewer note: Moving to a new place could mean the user no longer lives in Seattle, making the historical residence statement outdated, though the destination is unspecified.. 

### uncertain

- `009b7c99-b777-465a-bc3a-77f2d88c1318` → `16763588-459c-4dca-95ec-f7ff02c5a940`; sources: critical_harmful_transition; reviewer note: A walking boot may temporarily limit cycling and therefore affect the user's usual transportation preference, but the update does not establish that cycling is impossible or that the preference has changed.. 
- `009b7c99-b777-465a-bc3a-77f2d88c1318` → `7cf6d4c0-0201-46c2-9bcf-2cc7eb3d0c32`; sources: critical_harmful_transition; reviewer note: Wearing a walking boot may reflect a lower-limb injury or mobility issue that could affect standing or lifting, but the update does not establish that it changes the user's chronic lower back pain restrictions.. 
- `009b7c99-b777-465a-bc3a-77f2d88c1318` → `81a87a6e-c9b3-403b-971e-132b8388a983`; sources: critical_harmful_transition; reviewer note: Wearing a walking boot could affect the practicality of weekend wetland visits, but the update does not state reduced mobility, inability to visit, or any change in the user's values.. 

## Interpretation constraint

If the plausibly_affected category is substantial, interpret distractor-amplification claims conservatively because some hard negatives may not be truly unaffected. If clearly_unaffected cases dominate, the audit supplies model-based label-quality support for the failure analysis. This audit never changes benchmark gold labels or formal retrieval results.
