If(Test-Path -Path env\) {
    . env\Scripts\Activate.ps1
} else {
    python -m venv env
    . env\Scripts\Activate.ps1
    python -m pip install -r requirements.txt
}

If(Test-Path -Path storage\) {
    If(-Not (Test-Path -Path storage\choices\)) {
        New-Item -Path "storage\" -Name "choices" -ItemType "directory"
    }
    If(-Not (Test-Path -Path storage\weights\)) {
        New-Item -Path "storage\" -Name "weights" -ItemType "directory"
    }
    If(-Not (Test-Path -Path storage\theta\)) {
        New-Item -Path "storage\" -Name "theta" -ItemType "directory"
    }
} else {
        New-Item -Name "storage" -ItemType "directory"
        New-Item -Path "storage\" -Name "choices" -ItemType "directory"
        New-Item -Path "storage\" -Name "weights" -ItemType "directory"
        New-Item -Path "storage\" -Name "theta" -ItemType "directory"
}