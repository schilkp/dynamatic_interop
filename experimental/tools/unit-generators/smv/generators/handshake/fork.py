from generators.support.eager_fork_register_block import generate_eager_fork_register_block

def generate_fork (name, params):
    if "data_type" in params:
        return _generate_fork(name, params["size"], params["data_type"])
    else:
        return _generate_fork_dataless(name, params["size"])

def _generate_fork_dataless (name, size):
    return f"""
{generate_eager_fork_register_block(f"{name}__eager_fork_register_block")}

MODULE {name}(ins_valid, {", ".join([f"outs_ready_{n}" for n in range(size)])})
    {"\n    ".join([f"VAR inner_reg_block_{n} : {name}__eager_fork_register_block(ins_valid, outs_ready_{n}, backpressure);" for n in range(size)])}

    DEFINE any_block_stop := {" | ".join([f"inner_reg_block_{n}.block_stop" for n in range(size)])};
    DEFINE backpressure := ins_valid and any_block_stop;

    // output
    DEFINE ins_ready := !any_block_stop;
    {"\n    ".join([f"DEFINE outs_valid_{n} := inner_reg_block_{n}.outs_valid" for n in range(size)])}
    """

def _generate_fork (name, size, data_type):
    return f"""
    {_generate_fork_dataless(f"{name}__dataless_fork", size)}

MODULE {name}(ins, ins_valid, {", ".join([f"outs_ready_{n}" for n in range(size)])})
    VAR inner_fork : {name}__dataless_fork(ins_valid, {", ".join([f"outs_ready_{n}" for n in range(size)])});

    //output
    DEFINE ins_ready = inner_fork.ins_ready;
    {"\n    ".join([f"DEFINE outs_{n} := ins;" for n in range(size)])}
    {"\n    ".join([f"DEFINE outs_valid_{n} := inner_fork.outs_valid_{n};" for n in range(size)])}
    """


if __name__ == "__main__":
    print(generate_fork({"name": "test_fork", "size": 4, "data_type": None}))