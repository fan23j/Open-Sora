import subprocess

def run_command(command):
    try:
        result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return result.stdout.decode('utf-8').strip(), result.stderr.decode('utf-8').strip()
    except subprocess.CalledProcessError as e:
        return None, str(e)

def check_nvcc():
    print("Checking nvcc version... ", end="")
    output, error = run_command("nvcc --version")
    if output and "release 12.1" in output:
        print("OK")
        return True
    print("FAILED")
    return False

def check_python():
    print("Checking Python version... ", end="")
    output, error = run_command("python --version")
    if output and "Python 3.10." in output:
        print("OK")
        return True
    print("FAILED")
    return False

def check_pytorch():
    print("Checking PyTorch version... ", end="")
    output, error = run_command('python -c "import torch; print(torch.__version__)"')
    if output and "2.3.0" in output:
        print("OK")
        return True
    print("FAILED")
    return False

def check_cuda_version():
    print("Checking CUDA version... ", end="")
    output, error = run_command('python -c "import torch; print(torch.version.cuda)"')
    if output and "12.1" in output:
        print("OK")
        return True
    print("FAILED")
    return False

def check_apex():
    print("Checking Apex... ", end="")
    output, error = run_command('python -c "import apex"')
    if error:
        print("FAILED")
        return False
    print("OK")
    return True

def check_flash_attn():
    print("Checking Flash Attention... ", end="")
    output, error = run_command('python -c "import flash_attn"')
    if error:
        print("FAILED")
        return False
    print("OK")
    return True

def check_xformers():
    print("Checking xFormers... ", end="")
    output, error = run_command('python -m xformers.info')
    if output and "xFormers" in output:
        print("OK")
        return True
    print("FAILED")
    return False

def main():
    print("Starting environment check...\n")
    checks = [
        check_nvcc,
        check_python,
        check_pytorch,
        check_cuda_version,
        check_apex,
        check_flash_attn,
        check_xformers,
    ]

    all_checks_passed = True
    for check in checks:
        if not check():
            all_checks_passed = False

    if all_checks_passed:
        print("\nSUCCESS: All checks passed!")
    else:
        print("\nFAILED: Some checks did not pass.")

if __name__ == "__main__":
    main()
