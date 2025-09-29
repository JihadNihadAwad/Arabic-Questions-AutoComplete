package ai.project.gui;

import com.google.gson.JsonArray;
import com.google.gson.JsonElement;

import ai.project.ModelClient;
import javafx.application.Platform;
import javafx.collections.FXCollections;
import javafx.collections.ObservableList;
import javafx.fxml.FXML;
import javafx.fxml.Initializable;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.control.TableColumn;
import javafx.scene.control.TableView;
import javafx.scene.control.TextField;
import javafx.scene.control.cell.PropertyValueFactory;
import javafx.scene.input.KeyCode;
import javafx.scene.input.KeyEvent;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.net.URL;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.ResourceBundle;

public class MainController implements Initializable {

	@FXML
	private Button ann1;

	@FXML
	private Button ann2;

	@FXML
	private Button ann3;

	@FXML
	private Button lstm1;

	@FXML
	private Button lstm2;

	@FXML
	private Button lstm3;

	@FXML
	private TableView<Question> table;

	@FXML
	private Button tree1;

	@FXML
	private Button tree2;

	@FXML
	private Button tree3;

	@FXML
	private TextField txtInput;

	@FXML
	private Label lblWord;

	private ModelClient annClient;
	private ModelClient lstmClient;
	private ModelClient treeClient;

	private String suggestedWord = "";

	@Override
	public void initialize(URL location, ResourceBundle resources) {
		annClient = new ModelClient("http://localhost:5001"); // Replace with ANN server URL
		lstmClient = new ModelClient("http://localhost:5002"); // Replace with LSTM server URL
		treeClient = new ModelClient("http://localhost:5003"); // Replace with Decision Tree server URL

		txtInput.addEventFilter(KeyEvent.KEY_RELEASED, this::handleKeyReleased);
		txtInput.addEventFilter(KeyEvent.KEY_PRESSED, this::handleKeyPressed);

		// Add event handlers to buttons
        List<Button> allButtons = List.of(ann1, ann2, ann3, lstm1, lstm2, lstm3, tree1, tree2, tree3);
        for (Button button : allButtons) {
            button.setOnAction(event -> {
                // Append the text without a leading space
                String buttonText = button.getText();
                txtInput.appendText(buttonText + " ");

                // Fire a key released event to trigger prediction update
                txtInput.fireEvent(new KeyEvent(KeyEvent.KEY_RELEASED, "", "", KeyCode.UNDEFINED, false, false, false, false));
            });
        }

		initializeTable();
		loadQuestions(); // Load from example.txt

		// Load initial predictions
		loadInitialPredictions();
	}

	private void initializeTable() {
		TableColumn<Question, String> questionColumn = new TableColumn<>("Example Questions (Test Set)");
		questionColumn.setCellValueFactory(new PropertyValueFactory<>("questionText"));
		table.getColumns().add(questionColumn);
	}

	private void loadQuestions() {
	    try {
	        // Use getResourceAsStream to read directly from the JAR
	        InputStream is = getClass().getResourceAsStream("/examples.txt");
	        if (is == null) {
	            System.err.println("Could not find examples.txt in resources");
	            return;
	        }

	        // Use a BufferedReader to read lines efficiently
	        BufferedReader reader = new BufferedReader(new InputStreamReader(is));
	        ObservableList<Question> questions = FXCollections.observableArrayList();
	        String line;
	        while ((line = reader.readLine()) != null) {
	            questions.add(new Question(line));
	        }
	        reader.close(); // Close the reader

	        // Set the items in the TableView
	        table.setItems(questions);

	    } catch (IOException e) {
	        e.printStackTrace();
	    }
	}
	
	private void loadInitialPredictions() {
		updatePredictions("");
	}

	private void handleKeyReleased(KeyEvent event) {
		String inputText = txtInput.getText();
		updatePredictions(inputText);
	}

	private void handleKeyPressed(KeyEvent event) {
		if (event.getCode() == KeyCode.TAB && !suggestedWord.isEmpty()) {
			String currentText = txtInput.getText();
			String newText;

			if (currentText.isEmpty()) {
				newText = suggestedWord + " ";
			} else {
				String[] words = currentText.split("\\s+");
				String lastWord = words[words.length - 1];

				if (suggestedWord.startsWith(lastWord)) {
					int lastWordIndex = currentText.lastIndexOf(lastWord);
					newText = currentText.substring(0, lastWordIndex) + suggestedWord + " ";
				} else {
					newText = currentText + suggestedWord + " ";
				}
			}

			txtInput.setText(newText);
			txtInput.positionCaret(newText.length());
			event.consume();
		}
	}

	private void updatePredictions(String inputText) {
		List<Button> annButtons = List.of(ann1, ann2, ann3);
		List<Button> lstmButtons = List.of(lstm1, lstm2, lstm3);
		List<Button> treeButtons = List.of(tree1, tree2, tree3);

		Map<String, List<String>> allPredictions = new HashMap<>();

		try {
			JsonArray annPredictions = annClient.getPredictions(inputText, "ann", annButtons.size());
			List<String> annWords = extractWords(annPredictions);
			allPredictions.put("ann", annWords);
			Platform.runLater(() -> updateButtons(annButtons, annWords));
		} catch (IOException e) {
			Platform.runLater(() -> clearButtons(annButtons));
		}

		try {
			JsonArray lstmPredictions = lstmClient.getPredictions(inputText, "lstm", lstmButtons.size());
			List<String> lstmWords = extractWords(lstmPredictions);
			allPredictions.put("lstm", lstmWords);
			Platform.runLater(() -> updateButtons(lstmButtons, lstmWords));
		} catch (IOException e) {
			Platform.runLater(() -> clearButtons(lstmButtons));
		}

		try {
			JsonArray treePredictions = treeClient.getPredictions(inputText, "tree", treeButtons.size());
			List<String> treeWords = extractWords(treePredictions);
			allPredictions.put("tree", treeWords);
			Platform.runLater(() -> updateButtons(treeButtons, treeWords));
		} catch (IOException e) {
			Platform.runLater(() -> clearButtons(treeButtons));
		}

		updateHint(inputText, allPredictions);
	}

	private void updateButtons(List<Button> buttons, List<String> words) {
		for (int i = 0; i < buttons.size(); i++) {
			if (i < words.size()) {
				buttons.get(i).setText(words.get(i));
			} else {
				buttons.get(i).setText("");
			}
		}
	}

	private void clearButtons(List<Button> buttons) {
		for (Button button : buttons) {
			button.setText("NULL");
		}
	}

	private List<String> extractWords(JsonArray predictions) {
		List<String> words = new ArrayList<>();
		for (JsonElement prediction : predictions) {
			words.add(prediction.getAsJsonObject().get("word").getAsString());
		}
		return words;
	}

	private void updateHint(String inputText, Map<String, List<String>> allPredictions) {
		String[] words = inputText.split("\\s+");
		String lastWord = words.length > 0 ? words[words.length - 1] : "";

		// 1. Find shared prediction across all models
		suggestedWord = allPredictions.values().stream().reduce((list1, list2) -> {
			List<String> common = new ArrayList<>(list1);
			common.retainAll(list2);
			return common;
		}).orElse(new ArrayList<>()).stream().findFirst().orElse("");

		// 2. If no shared, prioritize LSTM, then ANN, then tree
		if (suggestedWord.isEmpty()) {
			if (allPredictions.containsKey("lstm") && !allPredictions.get("lstm").isEmpty()) {
				suggestedWord = allPredictions.get("lstm").get(0);
			} else if (allPredictions.containsKey("ann") && !allPredictions.get("ann").isEmpty()) {
				suggestedWord = allPredictions.get("ann").get(0);
			} else if (allPredictions.containsKey("tree") && !allPredictions.get("tree").isEmpty()) {
				suggestedWord = allPredictions.get("tree").get(0);
			}
		}

		// 3. Set lblWord text based on whether we're in the middle of a word or not
		if (!suggestedWord.isEmpty()) {
			if (!inputText.isEmpty() && suggestedWord.startsWith(lastWord)) {
				// Continue the current word
				Platform.runLater(() -> lblWord.setText(suggestedWord));
			} else {
				// Put a space and then the suggested word
				Platform.runLater(() -> lblWord.setText(" " + suggestedWord));
			}
		} else {
			Platform.runLater(() -> lblWord.setText(""));
		}
	}

	// Data class for TableView
	public static class Question {
		private final String questionText;

		public Question(String questionText) {
			this.questionText = questionText;
		}

		public String getQuestionText() {
			return questionText;
		}
	}
}