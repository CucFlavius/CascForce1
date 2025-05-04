using System.Net.Http.Headers;
using System.Text;
using System.Text.Json;
using System.Globalization;
using System.Text.RegularExpressions;
using System.Buffers;
using CsvHelper;
using CsvHelper.Configuration;
using OpenCL.Net;

namespace CascForce1;

class Program
{
    const string LookupUrl = "https://raw.githubusercontent.com/wowdev/wow-listfile/master/meta/lookup.csv";
    const string LookupFile = "lookup.csv";
    const string DictionaryFile = "dictionary.txt";
    const string GithubApiUrl = "https://api.github.com/repos/wowdev/wow-listfile/releases/latest";
    const string CommunityName = "community-listfile-withcapitals.csv";
    const int MaxMatches = 1024; // max number of matches to record on GPU

    static void Main(string[] args)
    {
        // BuildDictionaryFromListfile (
        //     CommunityName,
        //     DictionaryFile,
        //     "shaders\\"
        // );
        //
        // return;

        // 1) Prep: clean dictionary
        CleanUpDictionary(DictionaryFile);

        // 2) Parse args
        if (args.Length < 2)
        {
            Console.WriteLine("Usage: CascForce1 <mask> <maxWords>");
            return;
        }

        string mask = args[0].Replace('/', '\\').ToUpperInvariant();
        if (!int.TryParse(args[1], out int maxWords) || maxWords <= 0)
        {
            Console.WriteLine("Invalid maxWords. Must be a positive integer.");
            return;
        }

        // 3) Download lookup + diff-list
        DownloadLookup().GetAwaiter().GetResult();
        var skipIDs = DownloadDiffListFileAsync().GetAwaiter().GetResult();

        // 4) Load & filter lookup
        var allTargets = LoadLookupTargets().ToList();
        var filtered = allTargets.Where(t => !skipIDs.Contains(t.fileID)).ToList();
        var targetHashes = filtered
            .Select(t => ((ulong)t.hC << 32) | t.hB)
            .ToArray();
        uint targetCount = (uint)targetHashes.Length;
        Console.WriteLine($"Loaded {allTargets.Count} entries, {filtered.Count} after filtering.");

        // 5) Parse mask into prefix/suffix
        var parts = mask.Split('*');
        if (parts.Length != 2) throw new ArgumentException("Mask must contain exactly one '*'.");
        byte[] prefixBytes = Encoding.ASCII.GetBytes(parts[0]);
        byte[] suffixBytes = Encoding.ASCII.GetBytes(parts[1]);
        uint prefixLen = (uint)prefixBytes.Length;
        uint suffixLen = (uint)suffixBytes.Length;

        // 6) Load dictionary words
        if (!File.Exists(DictionaryFile))
        {
            Console.WriteLine($"Error: '{DictionaryFile}' not found.");
            return;
        }

        var words = File.ReadAllLines(DictionaryFile)
            .Select(w => w.Trim().ToUpperInvariant())
            .Where(w => w.Length > 0)
            .ToArray();
        int wCount = words.Length;
        Console.WriteLine($"Loaded {wCount} dictionary entries.");

        // 7) Compute max word length & flatten dictionary into fixed‐stride buffer
        int maxWordLen = words.Max(w => w.Length);
        byte[] dictData = new byte[wCount * maxWordLen];
        int[] wordLengths = new int[wCount];
        for (int i = 0; i < wCount; i++)
        {
            var b = Encoding.ASCII.GetBytes(words[i]);
            wordLengths[i] = b.Length;
            Array.Copy(b, 0, dictData, i * maxWordLen, b.Length);
        }

        // 8) Initialize OpenCL
        var context = InitializeOpenCL(maxWords, maxWordLen, prefixBytes, suffixBytes, out var queue, out var kernelSource, out var device);

        // 10) Compile OpenCL program & create kernel
        var kernel = CompileOpenCLProgram(context, device, kernelSource, dictData, wordLengths, prefixBytes, suffixBytes, targetHashes, out var dictBuf, out var lenBuf, out var prefixBuf, out var suffixBuf, out var targetBuf, out var counterBuf, out var indicesBuf);

        // 12) For each word‐count k, launch kernel
        bool anyFound = false;
        for (uint k = 1; k <= (uint)maxWords; k++)
        {
            // compute counts
            ulong baseCombos = 1;
            for (uint i = 0; i < k; i++) baseCombos *= (ulong)wCount;
            uint sepCombinations = k > 1 ? 1u << (int)(k - 1) : 1u;
            ulong total = baseCombos * sepCombinations;
            Console.WriteLine($"k={k}: total combos = {total:N0}");

            // reset matchCounter to 0
            uint zero = 0;
            Cl.EnqueueWriteBuffer(queue, counterBuf, Bool.True,
                IntPtr.Zero, new IntPtr(sizeof(uint)),
                new[] { zero }, 0, null, out _);

            // set kernel args
            uint arg = 0;
            Cl.SetKernelArg(kernel, arg++, dictBuf);
            Cl.SetKernelArg(kernel, arg++, lenBuf);
            Cl.SetKernelArg(kernel, arg++, (uint)wCount);
            Cl.SetKernelArg(kernel, arg++, (uint)maxWordLen);
            Cl.SetKernelArg(kernel, arg++, prefixBuf);
            Cl.SetKernelArg(kernel, arg++, prefixLen);
            Cl.SetKernelArg(kernel, arg++, suffixBuf);
            Cl.SetKernelArg(kernel, arg++, suffixLen);
            Cl.SetKernelArg(kernel, arg++, targetBuf);
            Cl.SetKernelArg(kernel, arg++, targetCount);
            Cl.SetKernelArg(kernel, arg++, k);
            Cl.SetKernelArg(kernel, arg++, baseCombos);
            Cl.SetKernelArg(kernel, arg++, sepCombinations);
            Cl.SetKernelArg(kernel, arg++, counterBuf);
            Cl.SetKernelArg(kernel, arg++, indicesBuf);

            // enqueue
            Cl.EnqueueNDRangeKernel(queue,
                kernel,
                1,
                null,
                new[] { new IntPtr((long)total) },
                null,
                0,
                null,
                out _);
            Cl.Finish(queue);

            // read back match count
            uint[] matchCountArr = new uint[1];
            Cl.EnqueueReadBuffer(queue, counterBuf, Bool.True,
                IntPtr.Zero, new IntPtr(sizeof(uint)),
                matchCountArr, 0, null, out _);
            uint matchCount = matchCountArr[0];
            if (matchCount == 0) continue;
            anyFound = true;

            // read back indices
            ulong[] matchIndices = new ulong[matchCount];
            Cl.EnqueueReadBuffer(queue, indicesBuf, Bool.True,
                IntPtr.Zero,
                new IntPtr(matchCount * sizeof(ulong)),
                matchIndices,
                0, null, out _);

            // reconstruct and print
            foreach (var idx in matchIndices)
            {
                // decode same as kernel
                ulong wordIdx = idx % baseCombos;
                int sepMask = (int)(idx / baseCombos);
                var sb = new StringBuilder();
                sb.Append(Encoding.ASCII.GetString(prefixBytes));
                int tmp = (int)wordIdx;
                int[] chosen = new int[k];
                for (int d = 0; d < k; d++)
                {
                    chosen[d] = tmp % wCount;
                    tmp /= wCount;
                }

                sb.Append(words[chosen[0]]);
                for (int d = 1; d < k; d++)
                {
                    if (((sepMask >> (d - 1)) & 1) != 0) sb.Append('_');
                    sb.Append(words[chosen[d]]);
                }

                sb.Append(Encoding.ASCII.GetString(suffixBytes));
                Console.WriteLine($"Match (k={k}): {sb}");
            }
        }

        if (!anyFound)
            Console.WriteLine("No matches found.");
        else
            Console.WriteLine("Scanning complete.");
    }

    static Kernel CompileOpenCLProgram(Context context, Device device, string kernelSource, byte[] dictData, int[] wordLengths,
        byte[] prefixBytes, byte[] suffixBytes, ulong[] targetHashes, out IMem dictBuf, out IMem lenBuf, out IMem prefixBuf,
        out IMem suffixBuf, out IMem targetBuf, out IMem counterBuf, out IMem indicesBuf)
    {
        var program = Cl.CreateProgramWithSource(context, 1, new[] { kernelSource }, null, out var error);
        // enable printf in the build
        error = Cl.BuildProgram(program,
            0,
            null,
            "-cl-std=CL1.2",
            null,
            IntPtr.Zero);

        if (error != ErrorCode.Success)
        {
            // if there’s a build error, grab the log so you can debug
            var log = Cl.GetProgramBuildInfo(program, device, ProgramBuildInfo.Log, out _);
            Console.WriteLine("BUILD LOG:\n" + log);
        }
        var kernel = Cl.CreateKernel(program, "BruteForce", out error);

        // 11) Create GPU buffers (read‐only for data, write‐only for matches)
        dictBuf = Cl.CreateBuffer(context,
            MemFlags.ReadOnly | MemFlags.CopyHostPtr,
            new IntPtr(dictData.Length),
            dictData,
            out _);
        lenBuf = Cl.CreateBuffer(context,
            MemFlags.ReadOnly | MemFlags.CopyHostPtr,
            new IntPtr(wordLengths.Length * sizeof(int)),
            wordLengths,
            out _);
        prefixBuf = Cl.CreateBuffer(context,
            MemFlags.ReadOnly | MemFlags.CopyHostPtr,
            new IntPtr(prefixBytes.Length),
            prefixBytes,
            out _);
        suffixBuf = Cl.CreateBuffer(context,
            MemFlags.ReadOnly | MemFlags.CopyHostPtr,
            new IntPtr(suffixBytes.Length),
            suffixBytes,
            out _);
        targetBuf = Cl.CreateBuffer(context,
            MemFlags.ReadOnly | MemFlags.CopyHostPtr,
            new IntPtr(targetHashes.Length * sizeof(ulong)),
            targetHashes,
            out _);
        counterBuf = Cl.CreateBuffer(context,
            MemFlags.ReadWrite,
            new IntPtr(sizeof(uint)),
            null,
            out _);
        indicesBuf = Cl.CreateBuffer(context,
            MemFlags.WriteOnly,
            new IntPtr(MaxMatches * sizeof(ulong)),
            null,
            out _);
        return kernel;
    }

    static Context InitializeOpenCL(int maxWords, int maxWordLen, byte[] prefixBytes, byte[] suffixBytes,
        out CommandQueue queue, out string kernelSource, out Device device)
    {
        ErrorCode error;
        Platform[] platforms = Cl.GetPlatformIDs(out error);
        var platform = platforms[0];
        Device[] devices = Cl.GetDeviceIDs(platform, DeviceType.Gpu, out error);
        for (int i = 0; i < devices.Length; i++)
        {
            var name = Cl.GetDeviceInfo(devices[i], DeviceInfo.Name, out error);
            Console.WriteLine($"Device {i}: {name}");
        }
        if (devices.Length == 0)
        {
            Console.WriteLine("No GPU devices found.");
        }
        Console.WriteLine($"Using device {0}: {Cl.GetDeviceInfo(devices[0], DeviceInfo.Name, out error)}");
        device = devices[0];
        var context = Cl.CreateContext(null, 1, new[] { device }, null, IntPtr.Zero, out error);
        queue = Cl.CreateCommandQueue(context, device, (CommandQueueProperties)0, out error);

        // 9) Build the kernel source dynamically
        kernelSource = $$"""
                         #pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
                         #pragma OPENCL EXTENSION cl_khr_printf           : enable
                         
                         #define MAX_WORDS {{maxWords}}
                         #define MAX_WORDLEN {{maxWordLen}}
                         #define MAX_BUF_SIZE ({{prefixBytes.Length}}+{{suffixBytes.Length}}+MAX_WORDLEN*MAX_WORDS+(MAX_WORDS-1))
                         #define MAX_MATCHES {{MaxMatches}}

                         __kernel void BruteForce(
                             __global const uchar* dictionaryData,
                             __global const int*   wordLengths,
                             const uint             wCount,
                             const uint             maxWordLen,
                             __global const uchar*  prefixData,
                             const uint             prefixLen,
                             __global const uchar*  suffixData,
                             const uint             suffixLen,
                             __global const ulong*  targetHashes,
                             const uint             targetCount,
                             const uint             k,
                             const ulong            baseCombos,
                             const uint             sepCombinations,
                             __global uint*         matchCounter,
                             __global ulong*        matchIndices
                         ) {
                             size_t idx = get_global_id(0);
                             if (idx >= baseCombos * sepCombinations) return;
                         
                             // split into word‐combo and sep mask
                             ulong wordIdx = idx % baseCombos;
                             uint sepMask = (uint)(idx / baseCombos);
                         
                             // build the candidate in local buffer
                             uchar buf[MAX_BUF_SIZE];
                             uint pos = 0;
                             for (uint i = 0; i < prefixLen; i++) buf[pos++] = prefixData[i];
                         
                             // choose words
                             uint chosen[MAX_WORDS];
                             ulong tmp = wordIdx;
                             for (uint d = 0; d < k; d++) {
                                 chosen[d] = (uint)(tmp % wCount);
                                 tmp /= wCount;
                             }
                         
                             // copy first word
                             uint wlen = wordLengths[chosen[0]];
                             for (uint i = 0; i < wlen; i++)
                                 buf[pos++] = dictionaryData[chosen[0] * maxWordLen + i];
                         
                             // subsequent words with optional underscore
                             for (uint d = 1; d < k; d++) {
                                 if (((sepMask >> (d - 1)) & 1) != 0)
                                     buf[pos++] = (uchar)'_';
                                 wlen = wordLengths[chosen[d]];
                                 for (uint i = 0; i < wlen; i++)
                                     buf[pos++] = dictionaryData[chosen[d] * maxWordLen + i];
                             }
                         
                             for (uint i = 0; i < suffixLen; i++) buf[pos++] = suffixData[i];
                         
                             // Jenkins hash
                             uint a = 0xdeadbeef + pos;
                             uint b = a;
                             uint c = a;
                             size_t i = 0;
                             #define ROT(x,r) ((x << r) | (x >> (32 - r)))
                             while (i + 12 <= pos) {
                                 a += (uint)buf[i]     + ((uint)buf[i+1] << 8)
                                   + ((uint)buf[i+2] << 16) + ((uint)buf[i+3] << 24);
                                 b += (uint)buf[i+4]   + ((uint)buf[i+5] << 8)
                                   + ((uint)buf[i+6] << 16) + ((uint)buf[i+7] << 24);
                                 c += (uint)buf[i+8]   + ((uint)buf[i+9] << 8)
                                   + ((uint)buf[i+10] << 16) + ((uint)buf[i+11] << 24);
                                 a -= c; a ^= ROT(c,4);  c += b;
                                 b -= a; b ^= ROT(a,6);  a += c;
                                 c -= b; c ^= ROT(b,8);  b += a;
                                 a -= c; a ^= ROT(c,16); c += b;
                                 b -= a; b ^= ROT(a,19); a += c;
                                 c -= b; c ^= ROT(b,4);  b += a;
                                 i += 12;
                             }
                             uint rem = pos - i;
                             if (rem >= 1)  a += buf[i];
                             if (rem >= 2)  a += (uint)buf[i+1] << 8;
                             if (rem >= 3)  a += (uint)buf[i+2] << 16;
                             if (rem >= 4)  a += (uint)buf[i+3] << 24;
                             if (rem >= 5)  b += buf[i+4];
                             if (rem >= 6)  b += (uint)buf[i+5] << 8;
                             if (rem >= 7)  b += (uint)buf[i+6] << 16;
                             if (rem >= 8)  b += (uint)buf[i+7] << 24;
                             if (rem >= 9)  c += buf[i+8];
                             if (rem >= 10) c += (uint)buf[i+9] << 8;
                             if (rem >= 11) c += (uint)buf[i+10] << 16;
                         
                             // final mixes
                             c ^= b; c -= ROT(b,14);
                             a ^= c; a -= ROT(c,11);
                             b ^= a; b -= ROT(a,25);
                             c ^= b; c -= ROT(b,16);
                             a ^= c; a -= ROT(c,4);
                             b ^= a; b -= ROT(a,14);
                             c ^= b; c -= ROT(b,24);
                         
                             ulong combined = (((ulong)c) << 32) | (ulong)b;
                         
                             // check against targets
                             for (uint t = 0; t < targetCount; t++)
                             {
                                 if (combined == targetHashes[t])
                                 {
                                     buf[pos] = 0;
                                     printf("- GPU MATCH! hash=0x%016llx path=\"%s\"\n", combined, buf);
                                     break;
                                 }
                             }
                         }

                         """;
        return context;
    }

    static void CleanUpDictionary(string dictionaryFile)
    {
        var words = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        if (File.Exists(dictionaryFile))
        {
            foreach (var line in File.ReadAllLines(dictionaryFile))
            {
                var w = line.Trim();
                if (w.Length > 0) words.Add(w);
            }
        }
        var sorted = words.OrderBy(w => w, StringComparer.OrdinalIgnoreCase);
        File.WriteAllLines(dictionaryFile, sorted);
    }


    static async Task DownloadLookup()
    {
        if (File.Exists(LookupFile)) return;
        using var client = new HttpClient();
        var data = await client.GetStringAsync(LookupUrl);
        await File.WriteAllTextAsync(LookupFile, data);
    }

    static async Task<HashSet<uint>> DownloadDiffListFileAsync()
    {
        if (File.Exists(CommunityName))
        {
            Console.WriteLine($"Loading cached {CommunityName}…");
            return ReadCommunityListFile(CommunityName);
        }
        using var client = new HttpClient();
        client.DefaultRequestHeaders.UserAgent.Add(new ProductInfoHeaderValue("GPUBruteForcer", "1.0"));
        var json = await client.GetStringAsync(GithubApiUrl);
        using var doc = JsonDocument.Parse(json);
        var assets = doc.RootElement.GetProperty("assets");
        string url = assets.EnumerateArray()
            .First(a => a.GetProperty("name").GetString().Equals(CommunityName, StringComparison.OrdinalIgnoreCase))
            .GetProperty("browser_download_url").GetString();
        Console.WriteLine($"Downloading {CommunityName}…");
        var csvData = await client.GetStringAsync(url);
        await File.WriteAllTextAsync(CommunityName, csvData);
        Console.WriteLine($"Saved to disk as {CommunityName}");
        return ReadCommunityListFile(CommunityName);
    }

    static HashSet<uint> ReadCommunityListFile(string path)
    {
        using var reader = new StreamReader(path);
        var cfg = new CsvConfiguration(CultureInfo.InvariantCulture) { Delimiter = ";", HasHeaderRecord = true };
        using var csv = new CsvReader(reader, cfg);
        if (!csv.Read() || !csv.ReadHeader()) return new HashSet<uint>();
        var set = new HashSet<uint>(); int total = 0;
        while (csv.Read()) { total++; if (csv.TryGetField(0, out uint id)) set.Add(id); }
        Console.WriteLine($"  → {total} rows; skipping {set.Count} unique IDs");
        return set;
    }

    static IEnumerable<(uint fileID, uint hC, uint hB)> LoadLookupTargets()
    {
        using var reader = new StreamReader(LookupFile);
        var cfg = new CsvConfiguration(CultureInfo.InvariantCulture) { Delimiter = ";", HasHeaderRecord = true };
        using var csv = new CsvReader(reader, cfg);
        if (!csv.Read() || !csv.ReadHeader()) yield break;
        while (csv.Read())
        {
            if (!csv.TryGetField(0, out uint fileID)) continue;
            var hex = csv.GetField(1);
            if (ulong.TryParse(hex, NumberStyles.HexNumber, CultureInfo.InvariantCulture, out ulong hv))
                yield return (fileID, (uint)(hv >> 32), (uint)hv);
        }
    }

    static void BuildDictionaryFromListfile(string listfileCsv, string dictionaryTxt, string prefixFilter)
    {
        var words = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        if (File.Exists(dictionaryTxt)) foreach (var line in File.ReadAllLines(dictionaryTxt)) { var w=line.Trim(); if(w.Length>0) words.Add(w); }
        using var reader = new StreamReader(listfileCsv);
        var cfg = new CsvConfiguration(CultureInfo.InvariantCulture) { Delimiter = ";", HasHeaderRecord = true };
        using var csv = new CsvReader(reader, cfg);
        if (!csv.Read() || !csv.ReadHeader()) throw new Exception($"Empty or malformed CSV: {listfileCsv}");
        var camelSplit = new Regex(@"(?<!^)(?=[A-Z])", RegexOptions.Compiled);
        while (csv.Read())
        {
            var path = csv.GetField(1);
            if (!path.StartsWith(prefixFilter, StringComparison.OrdinalIgnoreCase)) continue;
            path = path.Trim().Replace('/', '\\');
            var nameNoExt = Path.GetFileNameWithoutExtension(path);
            if (string.IsNullOrEmpty(nameNoExt)) continue;
            foreach (var part in nameNoExt.Split('_', StringSplitOptions.RemoveEmptyEntries))
                foreach (var sub in camelSplit.Split(part))
                    if (!string.IsNullOrEmpty(sub)) words.Add(sub);
        }
        File.WriteAllLines(dictionaryTxt, words.OrderBy(w=>w,StringComparer.OrdinalIgnoreCase));
        Console.WriteLine($"Built '{dictionaryTxt}' with {words.Count} unique words (prefix='{prefixFilter}').");
    }
}
