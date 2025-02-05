import pandas as pd

def get_loss_table(loss_fn):
    loss_df = pd.read_csv(loss_fn)[["fn", 
                                                "dev_loss", "dev_re_surface", "dev_re_ex_feats", "dev_kl_loss", 
                                                "test_loss", "test_re_surface", "test_re_ex_feats", "test_kl_loss"]]
    loss_df = loss_df.rename(columns={"fn": ""}).set_index("")
    loss_df = loss_df.rename(index={"no_ex.pt": "No EX", "ex_no_loss.pt": "EX No Loss", "ex_loss.pt": "EX Loss"})
    
    ltx = (loss_df.style.format("{:.2e}", 
            subset=["dev_loss", "dev_re_surface", "dev_re_ex_feats", "dev_kl_loss", 
                                "test_loss", "test_re_surface", "test_re_ex_feats", "test_kl_loss"])
                .to_latex(column_format="l" + "c" * len(loss_df.columns), hrules=True))
    ltx = ltx.replace("0.00e+00", "0")
    ltx = ltx.replace("& dev_loss & dev_re_surface & dev_re_ex_feats & dev_kl_loss & test_loss & test_re_surface & test_re_ex_feats & test_kl_loss \\\\\n &  &  &  &  &  &  &  &  \\\\\n\\midrule",
    r"""& \multicolumn{4}{c}{Validation} & \multicolumn{4}{c}{Test}\\
    & Loss & RE Surface & RE Return & KL & Loss & RE Surface & RE Return & KL \\
    \cmidrule(lr){2-5} \cmidrule(lr){6-9}""")
    return ltx