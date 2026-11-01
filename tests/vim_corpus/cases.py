"""Cases for scripts/diff_vim.py: (name, file name, start text, keys[, options]). Each must end in Normal mode.

Real Vim 9.1's results are in expected.json (scripts/diff_vim.py --record); tests/test_vim_engine.py checks
bash-lite's Vim against them.
"""

PY = '''import os
import sys

def train(model, data, lr=1e-3):
    """Train the model."""
    total = 0
    for batch in data:
        loss = model(batch)  # forward
        total += loss
    return total / len(data)


class Trainer:
    def __init__(self, epochs=10):
        self.epochs = epochs
        self.history = []

    def fit(self, x, y):
        result = train(x, y, lr=0.01)
        self.history.append(result)
        return {"loss": result, "epochs": self.epochs}
'''

TXT = '''The model is trained on a large dataset. It learns slowly at first.
Then the loss drops quickly! Does it overfit? We check the validation set.

Second paragraph starts here. It has two sentences.
  An indented line follows, with "quoted words" and (parenthesized text).

last-line my.var[0] = foo_bar(baz, qux); end
'''

CFG = '''epochs = 10
batch_size = 32
lr = 0.001
warmup = 0x1f
name = "resnet50"
layers = [64, 128, 256]
dropout=-5
'''

HTML = '''<div class="card">
  <p>Hello <b>world</b> and <i>friends</i></p>
  <ul>
    <li>one</li>
    <li>two</li>
  </ul>
</div>
'''

JSON = '''{"model": "bert", "layers": 12,
  "train": {"lr": 0.0003, "epochs": 4},
  "tags": ["nlp", "base"]}
'''

LIST = '''alpha
bravo
charlie
delta
echo
foxtrot
golf
hotel
india
juliet
'''

LONG = "\n".join(f"line {i:03d} of the long file" for i in range(1, 121)) + "\n"


def C(name, text, keys, fname=None, **opts):
    f = fname or {id(PY): "train.py", id(TXT): "notes.txt", id(CFG): "config.cfg", id(HTML): "page.html",
                  id(JSON): "data.json", id(LIST): "list.md", id(LONG): "long.txt"}.get(id(text), "file.txt")
    return (name, f, text, keys, opts) if opts else (name, f, text, keys)


CASES = []
add = CASES.append

# ---------------------------------------------------------------- motions
for i, k in enumerate(["w", "3w", "W", "b", "B", "e", "E", "ge", "gE", "0", "^", "$", "g_", "5l", "3h", "j", "4j",
                       "k", "G", "gg", "5G", "%", "}", "{", "2}", "fe", "Fe", "te", "Te", "f(;", "f,;,", "tl;",
                       "*", "#", "n", "+", "-", "_", "3|", "<CR>", " ", "<BS>", "H", "M", "L", "e;",
                       "$b", "2$", "fo2;", "10l"]):
    add(C(f"mot_txt_{i}", TXT, "2G5l" + k if not k.startswith("<") else "2G5l" + k))
for i, k in enumerate(["w", "b", "e", "ge", "W", "B", "E", "%", "f(", "t)", "}", "{", "*", "#", "gd", "4w", "$",
                       "^", "j$", "jj0w"]):
    add(C(f"mot_py_{i}", PY, "4G" + k))
for i, k in enumerate(["%", "f]%", "f{%", "$%", "w%"]):
    add(C(f"mot_match_{i}", PY, "21G" + k))
add(C("mot_wlast", TXT, "Gw"))
add(C("mot_elast", TXT, "G$e"))
add(C("mot_bfirst", TXT, "ggb"))
add(C("mot_wempty", TXT, "2G$w"))
add(C("mot_wempty2", TXT, "2Gww"))
add(C("mot_bempty", TXT, "4Gb"))
add(C("mot_jcol", TXT, "1G$jjj"))
add(C("mot_jwant", TXT, "1G20ljjjj"))
add(C("mot_Gcount", LIST, "3Gfl7G"))
add(C("mot_para_end", TXT, "6G}"))
add(C("mot_star_word", PY, "8G4w*"))
add(C("mot_search", PY, "/loss<CR>"))
add(C("mot_search2", PY, "/loss<CR>n"))
add(C("mot_searchN", PY, "G?self<CR>N"))
add(C("mot_search_off", PY, "/total/e<CR>"))
add(C("mot_search_wrap", PY, "G/import<CR>"))
add(C("mot_ctrl_d", LONG, "<C-d>", top=True))
add(C("mot_ctrl_d2", LONG, "<C-d><C-d>", top=True))
add(C("mot_ctrl_u", LONG, "60G<C-u>", top=True))
add(C("mot_ctrl_f", LONG, "<C-f>", top=True))
add(C("mot_ctrl_b", LONG, "100G<C-b>", top=True))
add(C("mot_H", LONG, "50GzzH", top=True))
add(C("mot_L", LONG, "50GzzL", top=True))
add(C("mot_M", LONG, "50GzzM", top=True))
add(C("mot_zt", LONG, "50Gzt", top=True))
add(C("mot_zb", LONG, "50Gzb", top=True))
add(C("mot_zz", LONG, "50Gzz", top=True))
add(C("mot_G_top", LONG, "G", top=True))
add(C("mot_jumpback", LONG, "50G<C-o>"))
add(C("mot_jumpfwd", LONG, "50G90G<C-o><C-o><Tab>"))
add(C("mot_backtick", LONG, "50G``"))
add(C("mot_quote", LONG, "50G5j''"))
add(C("mot_colon", LONG, ":42<CR>"))
add(C("mot_pct", LONG, "50%"))

# ---------------------------------------------------------------- operators
for i, k in enumerate(["dw", "d3w", "3dw", "de", "db", "dd", "3dd", "d$", "D", "d0", "d^", "dj", "dk", "dG", "dgg",
                       "d}", "d{", "dfe", "dte", "dFe", "dTe", "d%", "x", "3x", "X", "2X", "dl", "dh", "d2j",
                       "cwNEW<Esc>", "cWNEW<Esc>", "c$END<Esc>", "CEND<Esc>", "ccLINE<Esc>", "SLINE<Esc>",
                       "sZ<Esc>", "3sZ<Esc>", "ceX<Esc>", "cbX<Esc>", "yw", "yy", "y$", "yb", "3yy", "yj",
                       "ywP", "yyp", "yyP", "ddp", "ddP", "xp", "dwwP", "d/drops<CR>", "d?model<CR>",
                       "c/It<CR>X<Esc>", "y/loss<CR>P", "dE", "dW", "dB", "cEX<Esc>", "dge", "d2e"]):
    add(C(f"op_txt_{i}", TXT, "1G5w" + k))
for i, k in enumerate(["dw", "cwX<Esc>", "dW", "d$", "x", "D", "dd", "cc<Esc>", "yyjp", "dap", "dip", "yap",
                       "cipX<Esc>"]):
    add(C(f"op_eol_{i}", TXT, "1G$b" + k))
add(C("op_dw_lastword_line", TXT, "1G$bbdw"))
add(C("op_cw_blank", TXT, "5G0cwX<Esc>"))
add(C("op_cw_onchar", TXT, "1Gecw!<Esc>"))
add(C("op_dw_endfile", TXT, "G$bdw"))
add(C("op_d_para_excl", TXT, "4Gd}"))
add(C("op_d_para_mid", TXT, "4G3wd}"))
add(C("op_count_mult", LIST, "2d3w"))
add(C("op_dd_count_past", LIST, "8G5dd"))
add(C("op_J", LIST, "J"))
add(C("op_3J", LIST, "3J"))
add(C("op_gJ", LIST, "gJ"))
add(C("op_J_indent", PY, "4GJ"))
add(C("op_J_paren", TXT, "4GJ"))
add(C("op_J_period", TXT, "1GJ"))
add(C("op_tilde", TXT, "~~~"))
add(C("op_tilde_count", TXT, "5~"))
add(C("op_gUiw", TXT, "1G2wgUiw"))
add(C("op_guu", TXT, "guu"))
add(C("op_gUU", TXT, "gUU"))
add(C("op_g~~", TXT, "g~~"))
add(C("op_g~w", TXT, "g~w"))
add(C("op_gUap", TXT, "gUap"))
add(C("op_r", TXT, "rX"))
add(C("op_3r", TXT, "3rX"))
add(C("op_r_cr", TXT, "1G2wr<CR>"))
add(C("op_shift", PY, "4G>>"))
add(C("op_shift3", PY, "4G3>>"))
add(C("op_unshift", PY, "6G<<"))
add(C("op_shift_ip", PY, "4G>ip"))
add(C("op_shift_j", PY, "6G>j"))
add(C("op_eq", PY, "8G==", want=False))
add(C("op_eq_ip", PY, "4G=ip"))
add(C("op_ctrl_a", CFG, "<C-a>"))
add(C("op_ctrl_a10", CFG, "10<C-a>"))
add(C("op_ctrl_x", CFG, "2G5<C-x>"))
add(C("op_ctrl_a_hex", CFG, "4G<C-a>"))
add(C("op_ctrl_a_neg", CFG, "7G3<C-a>"))
add(C("op_ctrl_a_float", CFG, "3G<C-a>"))
add(C("op_ctrl_a_list", CFG, "6G<C-a>w<C-a>"))
add(C("op_ctrl_a_dot", CFG, "<C-a>j.j."))

# ---------------------------------------------------------------- text objects
for i, k in enumerate(["diw", "daw", "ciwX<Esc>", "yiwP", "diW", "daW", "dis", "das", "dip", "dap", "cisX<Esc>",
                       "2daw", "3diw", "d2aw", "viwd", "vawd", "vipd", "vapd"]):
    add(C(f"tobj_word_{i}", TXT, "1G6w" + k))
for i, k in enumerate(['di"', 'da"', 'ci"X<Esc>', "di(", "da(", "ci(X<Esc>", "dib", "dab", "yi(P"]):
    add(C(f"tobj_q_{i}", TXT, "5G3w" + k))
add(C("tobj_quote_before", TXT, "5G0di\""))
add(C("tobj_paren_before", TXT, "5G0di("))
add(C("tobj_bracket", TXT, "7Gfvdi["))
add(C("tobj_paren_args", PY, "4Gf(lci(m, d<Esc>"))
add(C("tobj_paren_on", PY, "4Gf(di("))
add(C("tobj_paren_close", PY, "4Gf)di("))
add(C("tobj_brace", PY, "21Gf{di{"))
add(C("tobj_brace_a", PY, "21Gf{da{"))
add(C("tobj_brace_ml", JSON, "2Gf{di{"))
add(C("tobj_bracket_ml", JSON, "3Gf[di["))
add(C("tobj_outer_brace", JSON, "2G2wdi{"))
add(C("tobj_tag_it", HTML, "2Gfwdit"))
add(C("tobj_tag_at", HTML, "2Gfwdat"))
add(C("tobj_tag_outer", HTML, "4Gfodit"))
add(C("tobj_tag_ul", HTML, "3Gdit"))
add(C("tobj_tag_cit", HTML, "4Gfncitzero<Esc>"))
add(C("tobj_ip_blank", TXT, "3Gdip"))
add(C("tobj_ap_blank", TXT, "3Gdap"))
add(C("tobj_ap_last", TXT, "7Gdap"))
add(C("tobj_2ap", TXT, "1Gd2ap"))
add(C("tobj_ip_py", PY, "5Gdip"))
add(C("tobj_ap_py", PY, "5Gdap"))
add(C("tobj_iw_space", TXT, "1Gwhdiw"))
add(C("tobj_aw_space", TXT, "1Gwhdaw"))
add(C("tobj_aw_end", TXT, "1G$daw"))
add(C("tobj_sentence2", TXT, "2G3wdas"))
add(C("tobj_sentence_i", TXT, "2G3wdis"))

# ---------------------------------------------------------------- insert mode
for i, k in enumerate(["ihi <Esc>", "ahi<Esc>", "Ihi<Esc>", "Ahi<Esc>", "onew<Esc>", "Onew<Esc>", "3ix<Esc>",
                       "3ahi<Esc>", "3onew<Esc>", "2Onew<Esc>", "ione<CR>two<Esc>", "ione two<C-w>x<Esc>",
                       "Aab cd<C-w><C-w>X<Esc>", "ione<C-u>X<Esc>", "Aend<BS><BS>X<Esc>", "i<BS><BS>X<Esc>",
                       "A<CR>new<Esc>", "otext<C-w><C-w>z<Esc>", "itrai<C-n><Esc>", "ilo<C-p><Esc>",
                       "ihi<Esc>u", "ihi<Esc>.", "Ahi<Esc>j.", "3ix<Esc>j.", "Ra<Esc>", "Rabc<BS><BS><Esc>",
                       "Rxyz<Esc>.", "gIx<Esc>", "ix<Esc>ggaY<Esc>gi!<Esc>", "a<C-r>.<Esc>", "yiwA <C-r>\"<Esc>",
                       "i<C-r>=6*7<CR><Esc>", "ione<Left><Left>X<Esc>", "ione<Esc>ugi2<Esc>"]):
    add(C(f"ins_txt_{i}", TXT, "1G2w" + k))
for i, k in enumerate(["onew = 1<Esc>", "Oabove<Esc>", "A<CR>x = 1<Esc>", "o<Esc>", "ox<CR>y<Esc>",
                       "A<CR><Esc>", "o<CR>z<Esc>", "ccx<Esc>", "Sx<Esc>", "o    pass<Esc>", "A<CR>pass<CR>y<Esc>",
                       "A<CR>return x<CR>y<Esc>", "otraining<C-t><Esc>", "A<C-d><Esc>", "o<Tab>x<Esc>",
                       "A<Tab>#<Esc>", "A<CR>a(<CR>b<Esc>"]):
    add(C(f"ins_py_{i}", PY, "4G" + k))
for i, k in enumerate(["onew = 1<Esc>", "A<CR>x<Esc>", "ccx<Esc>", "o<Esc>"]):
    add(C(f"ins_py8_{i}", PY, "8G" + k))
add(C("ins_ctrl_n_multi", PY, "Gotr<C-n><C-n><Esc>"))
add(C("ins_ctrl_n_back", PY, "Gotr<C-n><C-n><C-n><C-n><Esc>"))
add(C("ins_ctrl_p", PY, "Gose<C-p><Esc>"))
add(C("ins_tw_wrap", TXT, "Go" + "word " * 18 + "<Esc>"))

# ---------------------------------------------------------------- put, registers
for i, k in enumerate(['"ayy"ap', '"ayyj"Ayy"ap', '"adwj"ap', 'yyddj"0p', 'dd"1p', 'dddd"2p', 'dd"1p..',
                       'yiw"_dd"0p', 'yy3p', 'yiw3p', 'yiwjviwp', 'yyjVp', '"ayiw"byy"ap"bp', 'x"-p', 'yw$p',
                       'yyGp', 'ywgP', 'yyGgp', 'ddp', 'ddP', 'yiwbviwpviwp', '"Ayiw"ap']):
    add(C(f"reg_{i}", LIST, "2G" + k, regs=['"', "0", "1", "-", "a", "b"]))
add(C("reg_dot_insert", LIST, 'Afoo<Esc>j".p', regs=['"', "."]))
add(C("reg_block_put", LIST, "<C-v>2jlyGp", regs=['"', "0"]))
add(C("reg_block_P", LIST, "<C-v>2jlyG$p"))

# ---------------------------------------------------------------- undo, redo, dot
# (<Sync>: where a person would pause; Vim closes the undo block there)
for i, k in enumerate(["dd<Sync>u", "dd<Sync>dd<Sync>dd<Sync>u<Sync>u", "dd<Sync>u<Sync><C-r>", "x<Sync>x<Sync>x<Sync>u",
                       "x<Sync>x<Sync>x<Sync>u<Sync>u<Sync><C-r>", "ix<Esc><Sync>ay<Esc><Sync>u",
                       "cwX<Esc><Sync>u", "dd<Sync>j<Sync>dd<Sync>u", "3x<Sync>U", "x<Sync>j<Sync>x<Sync>U",
                       "x<Sync>x<Sync>U<Sync>U", "dw<Sync>.", "dw<Sync>.<Sync>.", "3dw<Sync>.", "dw<Sync>3.",
                       "cwX<Esc><Sync>w<Sync>.", "x<Sync>.", "dd<Sync>j<Sync>.", "A!<Esc><Sync>j<Sync>.<Sync>j<Sync>.",
                       ">><Sync>j<Sync>.", "dd<Sync>G<Sync>p<Sync>.", "dd<Sync>2.", "J<Sync>.", "~<Sync>.", "dt <Sync>.",
                       "u", "ix<Esc><Sync>u<Sync>u", "dd<Sync>u<Sync><C-r><Sync><C-r>", "dd<Sync>:earlier 1<CR>",
                       "x<Sync>x<Sync>g-", "x<Sync>x<Sync>g-<Sync>g-<Sync>g+", "ix<Esc><Sync>u<Sync>iy<Esc><Sync>g-",
                       "ix<Esc><Sync>u<Sync>iy<Esc><Sync>g-<Sync>g-<Sync>g+",
                       "qaA,<Esc>jq<Sync>2@a<Sync>u", "dd<Sync>p<Sync>u"]):
    add(C(f"undo_{i}", LIST, "2G" + k))

# ---------------------------------------------------------------- visual
for i, k in enumerate(["vjd", "vjy", "vjc!<Esc>", "Vjd", "Vjy", "Vj>", "Vj<", "vjJ", "vjgJ", "vj~", "vjU", "vju",
                       "veU", "vllr*", "vjo<Esc>", "vjoy", "v3l", "Vjd.", "vjd.", "v$d", "vwd", "viwd",
                       "vipd", "Vjj3>", "vjVd", "Vjvd", "vj<C-v>d", "vjx", "VjD", "vjD", "vjX", "VjJ",
                       "<C-v>jjd", "<C-v>jjlc*<Esc>", "<C-v>jjI# <Esc>", "<C-v>jjA;<Esc>", "<C-v>jj$A;<Esc>",
                       "<C-v>jjlr-", "<C-v>jjly", "<C-v>jjU", "<C-v>jjo<Esc>", "<C-v>jjO<Esc>", "<C-v>jjx",
                       "<C-v>jj$d", "<C-v>jjlI<<Esc>", "vjgvd", "vjv<Esc>gvd", "Vj:s/a/A/g<CR>",
                       "vj:normal A!<CR>", "Vjj:sort!<CR>", "<C-v>3j<C-a>", "Vjj<C-a>", "vjp", "yiwjVp",
                       "viw\"ay\"ap"]):
    add(C(f"vis_{i}", LIST, "2G" + k))
add(C("vis_block_short", TXT, "1G$<C-v>4jA!<Esc>"))
add(C("vis_block_I_short", TXT, "1G20l<C-v>4jI+<Esc>"))
add(C("vis_block_A_pad", TXT, "1G20l<C-v>4jA+<Esc>"))
add(C("vis_block_c_ml", PY, "6G4l<C-v>3jcXX<Esc>"))
add(C("vis_comment", PY, "6G<C-v>3jI# <Esc>"))
add(C("vis_uncomment", PY, "6G<C-v>3jI# <Esc><C-v>3jlld"))
add(C("vis_ip_V", TXT, "4Gvip"))
add(C("vis_ip_y", TXT, "4Gvipy"))

# ---------------------------------------------------------------- search and substitute
for i, k in enumerate([":s/the/THE/<CR>", ":s/a/A/g<CR>", ":%s/the/THE/g<CR>", ":%s/\\<it\\>/IT/g<CR>",
                       ":%s/\\vl(o+)ss/L\\1SS/g<CR>", ":%s/\\s\\+$//e<CR>", ":%s/\\(\\w\\+\\) \\(\\w\\+\\)/\\2 \\1/<CR>",
                       ":%s/is/&&/g<CR>", ":%s/\\./\\r/g<CR>", ":%s/model/\\u&/<CR>", ":%s/model/\\U&/<CR>",
                       ":2,4s/e/E/g<CR>", ":.,+2s/^/> /<CR>", ":%s/x/y/<CR>", ":%s//X/g<CR>",
                       "/loss<CR>:%s//LOSS/g<CR>", ":%s/a/b/gn<CR>", ":s/the/THE/<CR>j:s<CR>",
                       ":s/the/THE/<CR>j&", ":%s/the/THE/<CR>g&", ":g/the/d<CR>", ":v/the/d<CR>", ":g!/the/d<CR>",
                       ":g/^$/d<CR>", ":g/the/normal A!<CR>", ":g/^/m0<CR>", ":g/the/s/e/E/g<CR>",
                       ":%norm A,<CR>", ":2,3norm I- <CR>", ":%s/\\cTHE/x/g<CR>", ":%s/a\\|e/_/g<CR>",
                       ":%s/[aeiou]//g<CR>", ":%s/\\d\\+/N/g<CR>", ":%s/^\\(.*\\)$/[\\1]/<CR>", ":%s/ /\\t/<CR>",
                       ":%s/\\v(\\w+)\\s*\\=\\s*(\\w+)/\\2 = \\1/<CR>", ":%s/o\\zsn/N/g<CR>", ":%s/e\\ze /E/g<CR>",
                       ":%s/the/~/g<CR>", ":s/the/X/<CR>:%s/a/~/g<CR>", ":%s/\\(a\\)\\(b\\)\\?/[\\1\\2]/g<CR>",
                       ":%s/l\\{2,}/L/g<CR>", ":%s/s.\\{-}e/Q/g<CR>"]):
    add(C(f"sub_{i}", TXT, k))
for i, k in enumerate([":%s/a/A/gc<CR>yyny", ":%s/a/A/gc<CR>a", ":%s/a/A/gc<CR>nnq", ":%s/a/A/gc<CR>yl",
                       ":%s/the/X/c<CR>yny"]):
    add(C(f"sub_c_{i}", TXT, k))
for i, k in enumerate([":%s/= /=/<CR>", ":%s/\\v(\\w+) \\= (.*)/\\2 = \\1/<CR>", ":sort<CR>", ":sort!<CR>",
                       ":sort n<CR>", ":sort u<CR>", ":%!sort<CR>", ":2,4d<CR>", ":2,3m$<CR>", ":2,3t0<CR>",
                       ":1,2t$<CR>", ":$m0<CR>", ":3d<CR>", ":2,4y<CR>P", ":2,4>><CR>", ":2,4><CR>", ":3<CR>dd",
                       ":2;+1d<CR>", ":/lr/d<CR>", ":/lr/,/name/d<CR>", ":2,3j<CR>", ":2pu<CR>", ":0pu<CR>",
                       ":2,4normal A;<CR>", "/=<CR>ma:'a,$d<CR>"]):
    add(C(f"ex_{i}", CFG, k))
add(C("ex_sort_u_dup", "b\na\nb\nc\na\n", ":sort u<CR>", fname="dup.txt"))
add(C("ex_sort_n_mixed", "10 x\n9 y\nz\n-3 w\n100\n", ":sort n<CR>", fname="nums.txt"))
add(C("ex_sort_pat", "a 3\nb 1\nc 2\n", ":sort /\\a /<CR>", fname="p.txt"))

# ---------------------------------------------------------------- macros, marks, counts
for i, k in enumerate(["qaA,<Esc>jq3@a", "qaA,<Esc>jq@a@@", "qa0~jq9@a", "qaI\"<Esc>A\",<Esc>jq9@a",
                       "qaddq@a", "qayyjpq2@a", "qa@bqqbx@aq", "qaxq\"ap", "qaA!<Esc>jq:2,5normal @a<CR>",
                       "qq:s/a/A/<CR>jq3@q", "maGmb'add'bp", "majjd'a", "majjd`a", "majjy'aP", "ma3j:'a,.d<CR>",
                       "mxggdd`x", "ma`ap", "4Gma2G'ax", "/e<CR>cgnE<Esc>..", "/a<CR>dgn.."]):
    add(C(f"mac_{i}", LIST, "2G" + k))

# ---------------------------------------------------------------- options that change behaviour
add(C("opt_ic_search", TXT, ":set ic<CR>/THE<CR>"))
add(C("opt_scs_search", TXT, ":set ic scs<CR>/The<CR>"))
add(C("opt_et_shift", TXT, ":set et sw=2<CR>>>"))
add(C("opt_sw_shift", TXT, ":set sw=4 noet<CR>>>>>"))
add(C("opt_ai_o", TXT, ":set ai<CR>5Gox<Esc>"))
add(C("opt_nows", TXT, ":set nows<CR>G/model<CR>"))
add(C("opt_js", TXT, ":set nojs<CR>J"))

# ---------------------------------------------------------------- indentation: autoload/python.vim, indent/yaml.vim
PYI = 'def build(cfg):\n    name = ("m"\n        + "_v2")\nopt = "sgd"\n    if cfg.debug:\n    print(cfg)\n  return name\n'
PYI2 = "def f(x):\n    if x:\n        y = 1\n    return y\n"
PYI3 = "def f(a,\n      b):\n    x = foo(\n        1, 2)\n    return x\n"
YML = "a:\n  b: 1\nc:\n    d: 2\nitems:\n  - name: a\n    value: 1\n"
for i, (text, k) in enumerate([
        (PYI, "6GSpass<Esc>"), (PYI2, "4GSpass<Esc>"), (PYI2, "3Gccz = 2<Esc>"), (PYI2, "4GOz = 0<Esc>"),
        (PYI2, "3GA<CR>w = 2<CR><CR>x<Esc>"), (PYI, "3Gox<Esc>"), (PYI, "1Gox<Esc>"), (PYI, "7Gox<Esc>"),
        (PYI2, "3Goelse:<CR>z<Esc>"), (PYI2, "3Goelif y:<Esc>"), (PYI, "2Gox<Esc>"), (PYI3, "1Gox<Esc>"),
        (PYI3, "3Gox<Esc>"), (PYI3, "gg=G"), (PYI, "gg=G"), (PYI, "7G=="), (PYI2, "3God = {'a': 1}<Esc>"),
        ("try:\n    x = 1\n", "Goexcept E:<CR>pass<Esc>"), (PYI2, "2Gccj<Esc>")]):
    add(C(f"ind_py_{i}", text, k, fname="ind.py"))
for i, k in enumerate(["2Gccz: 3<Esc>", "4Gccz: 3<Esc>", "1Goz: 3<Esc>", "2Goz: 3<Esc>", "4GOz: 3<Esc>",
                       "6Goq: 2<Esc>", "7Go- name: b<Esc>", "gg=G", "Go- x<Esc>"]):
    add(C(f"ind_yaml_{i}", YML, k, fname="c.yaml"))
add(C("ind_txt_cc", "a\n    indented line\nb\n", "2Gccnew<Esc>", fname="n.txt"))
add(C("ind_md_cc", "a\n    indented line\nb\n", "2Gccnew<Esc>", fname="n.md"))
BRC = "int main() {\nint x = 1;\n      if (x) {\n  return 0;\n        }\nfor (int i = 0; i < 3; i++) {\nx += i;\n}\nreturn x;\n}\n"
JSN = '{\n"model": "gpt2",\n      "train": {\n  "epochs": 10,\n"lr": 0.001\n    },\n"tags": [\n"a",\n  "b"\n]\n}\n'
JS2 = "function train(model) {\n    for (const b of loader) {\n  const loss = step(b);\n          loss.backward();\n    }\n}\n"
for i, (fn, text, k) in enumerate([
        ("m.c", BRC, "3G=i{"), ("m.c", BRC, "gg=G"), ("c.json", JSN, "gg=G"), ("c.json", JSN, "4G=i{"),
        ("m.c", BRC, "3Gox;<Esc>"), ("t.js", JS2, "3G=i{"), ("t.js", JS2, "2Gox;<Esc>"), ("c.json", JSN, '3Go"x": 1<Esc>'),
        ("m.c", BRC, "9Go}<Esc>"), ("t.js", JS2, "5GOz;<Esc>"), ("m.c", BRC, "7Gox;<CR>}<Esc>")]):
    add(C(f"ind_brace_{i}", text, k, fname=fn))

# ---------------------------------------------------------------- 'undofile': history back after :bd, not after :e
for i, k in enumerate([":set undofile<CR>:w<CR>:bd<CR>:e undo.txt<CR>u", ":w<CR>:bd<CR>:e undo.txt<CR>u",
                       ":w<CR>:e<CR>u", ":set undofile<CR>:w<CR>:e<CR>u",
                       ":set undofile<CR>:w<CR>:bd<CR>:e undo.txt<CR>uu<C-r>"]):
    add(C(f"undofile_{i}", "a\nb\nc\n", "ggOz<Esc><Sync>" + k, fname="undo.txt"))

# ---------------------------------------------------------------- v$ then o: the line break stays selected
for i, k in enumerate(["v$obd", "v$od", "v$ooy$p", "v$ojd", "v$ohd", "v$ooly"]):
    add(C(f"vis_dollar_o_{i}", "one two three\nfour five\nsix\n", "1G5|" + k, fname="t.txt"))

# ---------------------------------------------------------------- long jumps: scroll to the edge or centre the cursor
LONGPY = "".join(f"def f{i}(x):\n    y = x + {i}\n    return y\n\n" for i in range(30))
for i, k in enumerate(["35G", "36G", "40G", "42G", ":set so=0<CR>36G", ":set so=0<CR>35G", "30Gzt20G", "30Gzt21G",
                       "30Gzt:set so=0<CR>20G", "30Gzt65G", "30Gzt64G", ":sp<CR>18G", ":sp<CR>17G",
                       ":sp<CR>:resize 7<CR>12G", "60Gzb40G", "60Gzb45G"]):
    add(C(f"jump_scroll_{i}", LONGPY, k, fname="long.py", top=True))
