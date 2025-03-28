#!/usr/bin/env python
# -*- coding: utf-8 -*-

print("Testing imports for SAE models...")

try:
    import models.sequential.SASRec_SAE_Attention
    print("Import successful: SASRec_SAE_Attention")
except Exception as e:
    print(f"Error importing SASRec_SAE_Attention: {e}")

try:
    import models.sequential.SASRec_SAE_Contrastive
    print("Import successful: SASRec_SAE_Contrastive")
except Exception as e:
    print(f"Error importing SASRec_SAE_Contrastive: {e}")

try:
    import models.sequential.SASRec_SAE_Hierarchical
    print("Import successful: SASRec_SAE_Hierarchical")
except Exception as e:
    print(f"Error importing SASRec_SAE_Hierarchical: {e}")

print("Import tests completed.") 