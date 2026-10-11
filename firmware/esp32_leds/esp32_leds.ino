// StoryForge LED strip: an ESP32 running FastLED, driven by storyforge/leds.py over USB serial.
//
// Protocol (115200 baud, one command per line):
//   "C,r,g,b,r,g,b,...\n"  90 values (r,g,b for 30 LEDs, 0-255) -> shows them, replies "OK"
// After boot it prints "READY". Opening the serial port resets the board, so the host waits
// for READY before sending. A malformed line is ignored and answered "ERR".
//
// The host already applies gamma and the brightness cap, so the sketch shows values as sent.
// Change DATA_PIN / LED_TYPE / COLOR_ORDER to match the wiring.

#include <FastLED.h>

#define NUM_LEDS    30
#define DATA_PIN    5
#define LED_TYPE    WS2812B
#define COLOR_ORDER GRB
#define BAUD        115200

CRGB leds[NUM_LEDS];
char line[NUM_LEDS * 12 + 16];  // "C," + up to "255," per value, with room to spare
size_t length = 0;

void handleLine() {
  line[length] = '\0';
  if (length < 2 || line[0] != 'C' || line[1] != ',') {
    Serial.println("ERR");
    return;
  }
  uint8_t values[NUM_LEDS * 3];
  int count = 0;
  char *cursor = line + 2;
  while (*cursor && count < NUM_LEDS * 3) {
    char *end;
    long value = strtol(cursor, &end, 10);
    if (end == cursor) break;
    values[count++] = (uint8_t)constrain(value, 0, 255);
    cursor = (*end == ',') ? end + 1 : end;
  }
  if (count != NUM_LEDS * 3) {
    Serial.println("ERR");
    return;
  }
  for (int i = 0; i < NUM_LEDS; i++) {
    leds[i] = CRGB(values[i * 3], values[i * 3 + 1], values[i * 3 + 2]);
  }
  FastLED.show();
  Serial.println("OK");
}

void setup() {
  Serial.begin(BAUD);
  FastLED.addLeds<LED_TYPE, DATA_PIN, COLOR_ORDER>(leds, NUM_LEDS);
  FastLED.setBrightness(255);
  FastLED.clear(true);
  delay(200);
  Serial.println("READY");
}

void loop() {
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\r') continue;
    if (c == '\n') {
      handleLine();
      length = 0;
    } else if (length < sizeof(line) - 1) {
      line[length++] = c;
    } else {
      length = 0;  // too long: drop it and wait for the next line
    }
  }
}
