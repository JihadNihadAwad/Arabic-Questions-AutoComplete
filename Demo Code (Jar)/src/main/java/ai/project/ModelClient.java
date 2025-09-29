package ai.project;

import com.google.gson.Gson;
import com.google.gson.JsonArray;
import com.google.gson.JsonObject;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;

public class ModelClient {

    private final String modelUrl;
    private final Gson gson = new Gson();

    public ModelClient(String modelUrl) {
        this.modelUrl = modelUrl;
    }

    public JsonArray getPredictions(String sentence, String modelType, int numPredictions) throws IOException {
        URL url = new URL(modelUrl + "/predict");
        HttpURLConnection con = (HttpURLConnection) url.openConnection();
        con.setRequestMethod("POST");
        con.setRequestProperty("Content-Type", "application/json");
        con.setDoOutput(true);

        String jsonInputString = String.format(
                "{\"model\": \"%s\", \"input\": \"%s\", \"num_predictions\": %d}",
                modelType, sentence, numPredictions);

        try (OutputStream os = con.getOutputStream()) {
            byte[] input = jsonInputString.getBytes(StandardCharsets.UTF_8);
            os.write(input, 0, input.length);
        }

        int responseCode = con.getResponseCode();
        if (responseCode == HttpURLConnection.HTTP_OK) {
            try (BufferedReader br = new BufferedReader(
                    new InputStreamReader(con.getInputStream(), StandardCharsets.UTF_8))) {
                StringBuilder response = new StringBuilder();
                String responseLine;
                while ((responseLine = br.readLine()) != null) {
                    response.append(responseLine.trim());
                }
                JsonObject jsonResponse = gson.fromJson(response.toString(), JsonObject.class);
                return jsonResponse.getAsJsonArray("output");
            }
        } else {
            throw new IOException("HTTP error code: " + responseCode);
        }
    }
}