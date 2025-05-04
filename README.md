# CascForce1

A simple GPU bruteforcer for shader file names in WoW Classic.  
Leverages OpenCL to massively parallelize Jenkins‐hash checks against a community‐curated listfile.

---

## Features

- **GPU-accelerated** brute-force search of file‐name combinations  
- Uses Jenkins hash matching against the 
- Automatically downloads and diffs against the latest community listfile release [WoW listfile](https://github.com/wowdev/wow-listfile)  
- Cleans and builds a dictionary of name fragments for mask-based searching  

---

## Prerequisites

- [.NET 6.0 SDK or later](https://dotnet.microsoft.com/download)  
- OpenCL-capable GPU and drivers (NVIDIA, AMD, or Intel)  
- Internet access for initial lookup and community listfile download  
- `lookup.csv` (automatically downloaded)  
- `dictionary.txt` (generated or provided)  

---

## Installation

1. **Clone this repository**  
   ```bash
   git clone https://github.com/yourusername/CascForce1.git
   cd CascForce1
   dotnet restore
   dotnet build -c Release

## Usage
- Syntax:
`CascForce1 <mask> <maxWords>`

- Example:
`CascForce1 "shaders/hull/dx_6_0/*.bls" 4`

- Dictionary Generation (Optional)
If you need to build a fresh dictionary from a local listfile:
    ```// in Program.cs, uncomment:
    BuildDictionaryFromListfile(
        "community-listfile-withcapitals.csv",
        "dictionary.txt",
        "shaders\\"
    );
    return;
