## Analysis scripts

This directory includes implementations for the analyses in the paper. Note that the codes will not be working as is. For example, we didn't include `create_model_and_transforms` in this first release version. However, you will easily fix the code by using the huggingface wrapper, which can be found in the [example.ipynb](../example.ipynb)

### HierarCaps

- For more details of HierarCaps, please check the official repository and paper first: [TAU-VAILab/hierarcaps](https://github.com/TAU-VAILab/hierarcaps)
- Check `eval_hcaps.py` for the implementation of our HierarCaps retrieval.
- Use `--with_uncertainty_with_root` option for reproducing Table 2.

### ImageNet Zero-shot Classification Prompt Re-weighting

- Check `unc_zeroshot_prompts.py` for the implemtation of prompt re-weighting using ProLIP.
- See `bprw` function for the detailed implementation of BPRW
