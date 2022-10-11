#include <iostream>
#include <string>

int main()
{
        typedef struct  WAV_HEADER
{
    // RIFF Chunk
    uint8_t         Chunk_ID[4];        // RIFF
    uint32_t        Chunk_data_Size;      // RIFF Chunk data Size
    uint8_t         RIFF_TYPE_ID[4];        // WAVE
    // format sub-chunk
    uint8_t         Chunk1_ID[4];         // fmt
    uint32_t        chunk1_data_Size;  // Size of the format chunk
    uint16_t        Format_Tag;    //  format_Tag 1=PCM
    uint16_t        Num_Channels;      //  1=Mono 2=Sterio
    uint32_t        Sample_rate;  // Sampling Frequency in (44100)Hz
    uint32_t        byte_rate;    // Byte rate
    uint16_t        block_Align;     // 4
    uint16_t        bits_Per_Sample;  // 16
    /* "data" sub-chunk */
    uint8_t         chunk2_ID[4]; // data
    uint32_t        chunk2_data_Size;  // Size of the audio data
} obj;
obj header;
    const char* filePath;
    std::string input;
    {
        std::cout << "Enter the wave file name: ";
        std::cin >> input;
        cin.get();
        filePath = input.c_str();
    }
    FILE* fp = fopen(filePath, "r");
    if (fp == NULL)
    {
        fprintf(stderr, " file cannot be open %s \n", filePath);
    }
    {
         size_t num = fread(&header, 1, sizeof(header), fp);
    std::cout << "Header size " << num << " bytes." << endl;
   return 0;
    }
}