#include "pgm_io.h"

#include <cctype>
#include <fstream>

namespace pgm {
namespace {

// Reads the next whitespace-delimited integer token, skipping '#' comments.
bool ReadInt(std::istream& in, int* value) {
  int c = in.get();
  while (in && (isspace(c) || c == '#')) {
    if (c == '#') {
      while (in && c != '\n') c = in.get();
    }
    c = in.get();
  }
  if (!in || !isdigit(c)) return false;
  int result = 0;
  while (in && isdigit(c)) {
    result = result * 10 + (c - '0');
    c = in.get();
  }
  *value = result;  // The single delimiter after the token is consumed.
  return true;
}

}  // namespace

bool Read(const std::string& path, Image* image, std::string* error) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    *error = "cannot open " + path;
    return false;
  }
  char magic[2];
  in.read(magic, 2);
  if (!in || magic[0] != 'P' || magic[1] != '5') {
    *error = path + ": not a binary PGM (P5)";
    return false;
  }
  int width = 0, height = 0, maxval = 0;
  if (!ReadInt(in, &width) || !ReadInt(in, &height) ||
      !ReadInt(in, &maxval) || width <= 0 || height <= 0 || maxval != 255) {
    *error = path + ": bad header (only 8-bit PGM supported)";
    return false;
  }
  image->width = width;
  image->height = height;
  image->pixels.resize(static_cast<size_t>(width) * height);
  in.read(reinterpret_cast<char*>(image->pixels.data()),
          static_cast<std::streamsize>(image->pixels.size()));
  if (in.gcount() != static_cast<std::streamsize>(image->pixels.size())) {
    *error = path + ": truncated pixel data";
    return false;
  }
  return true;
}

bool Write(const std::string& path, int width, int height,
           const uint8_t* pixels) {
  std::ofstream out(path, std::ios::binary);
  if (!out) return false;
  out << "P5\n" << width << " " << height << "\n255\n";
  out.write(reinterpret_cast<const char*>(pixels),
            static_cast<std::streamsize>(width) * height);
  return static_cast<bool>(out);
}

}  // namespace pgm
