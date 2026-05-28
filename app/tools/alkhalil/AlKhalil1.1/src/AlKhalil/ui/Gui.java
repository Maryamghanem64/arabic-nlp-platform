/*
 * Gui.java
 *
 *
 */
/* ALKHALIL MORPHO SYS -- An open source programm.
 *
 * Copyright (C) 2010.
 *
 * This program is free software, distributed under the terms of
 * the GNU General Public License Version 3. For more informations see web site at :
 * http://www.gnu.org/licenses/gpl.txt
 */
package AlKhalil.ui;

import java.awt.*;
import java.awt.Desktop;
import java.awt.event.*;
import javax.swing.*;
import java.util.*;
import java.util.List;
import java.io.*;
import javax.swing.border.Border;
import AlKhalil.token.*;
import AlKhalil.result.tables.*;
import AlKhalil.result.SearchResults;
import AlKhalil.analyse.*;
import AlKhalil.*;
import AlKhalil.result.TextIndexOutput;

/**
 *This is the Graphical user interface class which allows the user to test Alkhalil.
 * it contains a JTextPane to insert the input text and many JMenuItems for all Alkhalil options.
 *
 *
 */
public class Gui extends JFrame {

    private JMenuBar menu;
    private JMenu file, help, userguide, tools;
    private JMenuItem open, save_as, close, analyse, show, save, userguideAr, userguideEn, about;
    public static JMenuItem setting, select, search, textIndex;
    private JTextPane inputText;
    public static Analyzer analyzer;
    public static TextIndexOutput formTextInd = new TextIndexOutput();

    /** Creates a new instance of Gui */
    public Gui() {
        super("»—‰«„Ã  «·Œ·Ì· «·’—›Ì " + " AlKhalil Morpho Sys");
        EventManager evt = new EventManager();
        setSize(600, 500);
        new Settings();
        analyzer = new Analyzer();
        //creat the menu bar
        menu = new JMenuBar();
        setJMenuBar(menu);//add the menu bar to the frame

        menu.applyComponentOrientation(ComponentOrientation.getOrientation(new Locale("ar")));

        //creat the menu file and his options open, save, save as and close
        file = new JMenu("„·›");
        file.applyComponentOrientation(ComponentOrientation.getOrientation(new Locale("ar")));
        file.setFont(new java.awt.Font("Traditional Arabic", 1, 24));
        menu.add(file);


        open = new JMenuItem("› Õ");
        open.setFont(new java.awt.Font("Traditional Arabic", 1, 20));
        open.applyComponentOrientation(ComponentOrientation.getOrientation(new Locale("ar")));
        file.add(open);
        file.addSeparator();
        open.addActionListener(evt);

        save_as = new JMenuItem("Õ›Ÿ »«”„");
        save_as.setFont(new java.awt.Font("Traditional Arabic", 1, 20));
        save_as.applyComponentOrientation(ComponentOrientation.getOrientation(new Locale("ar")));
        file.add(save_as);
        file.addSeparator();
        save_as.addActionListener(evt);
        close = new JMenuItem("Œ—ÊÃ");
        close.setFont(new java.awt.Font("Traditional Arabic", 1, 20));
        close.applyComponentOrientation(ComponentOrientation.getOrientation(new Locale("ar")));
        close.addActionListener(evt);
        file.add(close);


        //creat the option analyse
        tools = new JMenu("√œÊ« ");
        tools.applyComponentOrientation(ComponentOrientation.getOrientation(new Locale("ar")));
        tools.setFont(new java.awt.Font("Traditional Arabic", 1, 24));

        analyse = new JMenuItem(" Õ·Ì·");
        analyse.setFont(new java.awt.Font("Traditional Arabic", 1, 20));
        analyse.applyComponentOrientation(ComponentOrientation.getOrientation(new Locale("ar")));
        tools.add(analyse);
        analyse.addActionListener(evt);
        tools.addSeparator();

        show = new JMenuItem("⁄—÷ ‰ «∆Ã «· Õ·Ì·");
        show.setFont(new java.awt.Font("Traditional Arabic", 1, 20));
        show.applyComponentOrientation(ComponentOrientation.getOrientation(new Locale("ar")));
        tools.add(show);
        show.addActionListener(evt);
        show.setEnabled(false);
        tools.addSeparator();

        save = new JMenuItem("Õ›Ÿ «·‰ «∆Ã");
        save.setFont(new java.awt.Font("Traditional Arabic", 1, 20));
        save.applyComponentOrientation(ComponentOrientation.getOrientation(new Locale("ar")));
        tools.add(save);
        save.addActionListener(evt);
        save.setEnabled(false);
        tools.addSeparator();
        select = new JMenuItem("«Œ Ì«— »⁄÷ «·‰ «∆Ã");
        select.setFont(new java.awt.Font("Traditional Arabic", 1, 20));
        select.applyComponentOrientation(ComponentOrientation.getOrientation(new Locale("ar")));
        tools.add(select);
        select.addActionListener(evt);
        select.setEnabled(false);
        tools.addSeparator();
        search = new JMenuItem("»ÕÀ ›Ì «·‰ «∆Ã");
        search.setFont(new java.awt.Font("Traditional Arabic", 1, 20));
        search.applyComponentOrientation(ComponentOrientation.getOrientation(new Locale("ar")));
        tools.add(search);
        search.addActionListener(evt);
        search.setEnabled(false);
        tools.addSeparator();
        textIndex = new JMenuItem("⁄—÷ ‰ «∆Ã «·›Â—”…");
        textIndex.setFont(new java.awt.Font("Traditional Arabic", 1, 20));
        textIndex.applyComponentOrientation(ComponentOrientation.getOrientation(new Locale("ar")));
        tools.add(textIndex);
        textIndex.addActionListener(evt);
        textIndex.setEnabled(false);
        tools.addSeparator();

        setting = new JMenuItem("·ÊÕ… «· Õﬂ„");
        setting.setFont(new java.awt.Font("Traditional Arabic", 1, 20));
        setting.applyComponentOrientation(ComponentOrientation.getOrientation(new Locale("ar")));
        tools.add(setting);
        setting.addActionListener(evt);


        menu.add(tools);


        //creat the menu file and the options open, save, save as and close
        help = new JMenu("„”«⁄œ…");
        help.applyComponentOrientation(ComponentOrientation.getOrientation(new Locale("ar")));
        help.setFont(new java.awt.Font("Traditional Arabic", 1, 24));
        menu.add(help);

        userguide = new JMenu("œ·Ì· «·«” ⁄„«·");
        userguide.applyComponentOrientation(ComponentOrientation.getOrientation(new Locale("ar")));
        userguide.setFont(new java.awt.Font("Traditional Arabic", 1, 20));
        help.add(userguide);

        userguideAr = new JMenuItem("⁄—»Ì");
        userguideAr.setFont(new java.awt.Font("Traditional Arabic", 1, 20));
        userguideAr.applyComponentOrientation(ComponentOrientation.getOrientation(new Locale("ar")));
        userguide.add(userguideAr);
        userguideAr.addActionListener(evt);
        userguide.addSeparator();
        userguideEn = new JMenuItem("«‰Ã·Ì“Ì");
        userguideEn.setFont(new java.awt.Font("Traditional Arabic", 1, 20));
        userguideEn.applyComponentOrientation(ComponentOrientation.getOrientation(new Locale("ar")));
        userguide.add(userguideEn);
        userguideEn.addActionListener(evt);
        help.addSeparator();

        about = new JMenuItem("ÕÊ· «·»—‰«„Ã");
        about.setFont(new java.awt.Font("Traditional Arabic", 1, 20));
        about.applyComponentOrientation(ComponentOrientation.getOrientation(new Locale("ar")));
        help.add(about);
        about.addActionListener(evt);
        inputText = new JTextPane();
        inputText.setComponentOrientation(ComponentOrientation.RIGHT_TO_LEFT);
        inputText.setFont(new java.awt.Font("Traditional Arabic", 1, 32));


        JScrollPane js = new JScrollPane(inputText);
        getContentPane().add(js);
        //inputText.setText(" ");
        addWindowListener(new WindowCloser());
        show();
    }

    private class EventManager implements ActionListener {

        //--------------------------------------------------------------------------------
        public void analyseActionPerformed() throws Exception {

            Thread t = new Thread() {

                public void run() {
                    long startTime = Calendar.getInstance().getTimeInMillis();
                    try {
                        if (Settings.dbchoice) {
                            analyzer.VRoots = analyzer.db.LoadRoots("db/AllRoots2.txt");
                            analyzer.NRoots = analyzer.db.LoadRoots("db/AllRoots2.txt");
                        } else {
                            analyzer.VRoots = analyzer.db.LoadRoots("db/AllRoots1.txt");
                            analyzer.NRoots = analyzer.db.LoadRoots("db/AllRoots1.txt");
                        }
                    } catch (Exception ex) {
                        ex.printStackTrace();
                    }
                    analyzer.allResults = new HashMap();
                    analyse.setEnabled(false);

                    String textToBeAnalysed = inputText.getSelectedText() != null ? inputText.getSelectedText() : inputText.getText();

                    Tokens tokens = new Tokens(textToBeAnalysed);

                    List unvoweledTokens = tokens.getUnvoweledTokens();

                    List normalizedTokens = tokens.getNormalizedTokens();


                    //------------
                    final JFrame f = new JFrame("Õ«·… «· Õ·Ì·");
                    //f.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
                    Container content = f.getContentPane();
                    JProgressBar progressBar = new JProgressBar();

                    content.add(progressBar, BorderLayout.NORTH);


                    f.setSize(300, 100);
                    f.setVisible(true);
                    //------
                    f.setResizable(false);

                    for (int i = 0; i < unvoweledTokens.size(); i++) {// process all tokens

                        String unvoweledWord = (String) unvoweledTokens.get(i);

                        String normalizedWord = (String) normalizedTokens.get(i);

                        progressBar.setStringPainted(true);
                        Border border = BorderFactory.createTitledBorder("«·‰’ ﬁÌœ «· Õ·Ì·....");
                        progressBar.setBorder(border);
                        float o = 100 * (i + 1) / unvoweledTokens.size();

                        progressBar.setValue((int) o);

                        List result = new LinkedList();
                        try {
                            result = analyzer.Analyze(normalizedWord, unvoweledWord);
                        } catch (Exception ex) {
                            ex.printStackTrace();
                        }
                        HashMap sol = new HashMap();
                        sol.put(normalizedWord, result);
                        analyzer.allResults.put(i, sol);




                    }


                    Settings.upDateResults();

                    long endTime = Calendar.getInstance().getTimeInMillis();
                    //System.out.println(endTime - startTime);


                    show.setEnabled(true);
                    save.setEnabled(true);
                    select.setEnabled(true);
                    search.setEnabled(true);
                    textIndex.setEnabled(true);

                    Toolkit.getDefaultToolkit().beep();
                    f.dispose();


                    //
                    JOptionPane pane = new JOptionPane("  „  Õ·Ì· «·‰’ «·„œŒ·\n");
                    pane.applyComponentOrientation(ComponentOrientation.getOrientation(new Locale("ar")));
                    Object[] options = new String[]{"⁄—÷ «·‰ «∆Ã", "≈€·«ﬁ"};
                    pane.setOptions(options);
                    JDialog dialog = pane.createDialog(new JFrame(), " ‰»ÌÂ");
                    dialog.applyComponentOrientation(ComponentOrientation.getOrientation(new Locale("ar")));
                    dialog.show();
                    Object obj = pane.getValue();
                    int result = -1;
                    for (int k = 0; k < options.length; k++) {
                        if (options[k].equals(obj)) {
                            result = k;
                        }
                    }
                    if (result == 0) {
                        HtmlTables.showResults();
                    }

                    //

                    //f.setVisible(true);
                    // f.setSize(300, 200);
                    JButton sh = new JButton("Â·  Êœ ⁄—÷ «·‰ «∆Ãø");

                    f.add(sh);

                    sh.addActionListener(new java.awt.event.ActionListener() {

                        public void actionPerformed(java.awt.event.ActionEvent evt) {
                            HtmlTables.showResults();
                            f.setVisible(false);
                        }
                    });


                    Border border = BorderFactory.createTitledBorder(" „  Õ·Ì· «·‰’");
                    progressBar.setBorder(border);


                    analyse.setEnabled(true);
                }
            };
            t.start();

            Thread indexThread = new Thread() {

                @Override
                public void run() {
                    new TextIndex().TextIndex(inputText.getText().toString());
                }
            };
            indexThread.start();

        }

        private void saveFileAs() {
            JFileChooser jfc = new JFileChooser();
            jfc.setDialogTitle("Save As...");
            jfc.setApproveButtonText("Save");
            jfc.setApproveButtonToolTipText("Here");

            int resultat = jfc.showOpenDialog(null);
            if (resultat == JFileChooser.APPROVE_OPTION) {
                writeFile(jfc.getSelectedFile());
            }
        }

        private void openFile() {
            JFileChooser jfc = new JFileChooser();

            int resultat = jfc.showOpenDialog(null);
            if (resultat == JFileChooser.APPROVE_OPTION) {

                readFile(jfc.getSelectedFile());
            } else {
                //fichierCourant = null;
            }
        }

        private void readFile(File fichier) {
            String ligne;

            StringBuffer buf = new StringBuffer();
            try {
                inputText.setText("");

                BufferedReader in = new BufferedReader(
                        new InputStreamReader(
                        new FileInputStream(fichier), "Cp1256"));


                while ((ligne = in.readLine()) != null) {
                    buf.append(ligne + (char) '\n');
                }
                inputText.setText(buf.toString());
            } catch (FileNotFoundException e) {
                e.printStackTrace();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }

        private void writeFile(File fichier) {
            try {
                OutputStreamWriter out = new OutputStreamWriter(
                        new FileOutputStream(fichier), "Cp1256");
                out.write(inputText.getText());
                out.close();
            } catch (FileNotFoundException e) {
                e.printStackTrace();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }

        public void actionPerformed(ActionEvent e) {
            if (e.getSource() == open) {
                openFile();
            }
            // if(e.getSource()==save){}
            if (e.getSource() == save_as) {
                saveFileAs();
            }
            if (e.getSource() == close) {
                System.exit(0);
            }
            if (e.getSource() == analyse) {

                try {
                    analyseActionPerformed();
                } catch (Exception ex) {
                    ex.printStackTrace();
                }


            }
            if (e.getSource() == show) {

                HtmlTables.showResults();


            }
            if (e.getSource() == save) {

                CsvTables.traitement();


            }
            if (e.getSource() == userguideAr) {

                try {
                    Desktop.getDesktop().open(new File("doc/ArabicUserguide.htm"));
                } catch (Exception el) {
                    el.printStackTrace();
                }


            }
            if (e.getSource() == userguideEn) {

                try {
                    Desktop.getDesktop().open(new File("doc/EnglishUserguide.htm"));
                } catch (Exception el) {
                    el.printStackTrace();
                }

            }
            if (e.getSource() == about) {

                About.main(null);


                //
            }
            if (e.getSource() == setting) {
                new Settings().setVisible(true);
                setting.setEnabled(false);
                //
            }
            if (e.getSource() == select) {
                CsvTables.showResults();
                //
            }
            if (e.getSource() == search) {
                //CsvTables.showResults();
                new SearchResults().setVisible(true);
                //
            }
            if (e.getSource() == textIndex) {
                formTextInd.setVisible(false);
                formTextInd.dispose();
                formTextInd = new TextIndexOutput();
                // new TextIndex().TextIndex(inputText.getText().toString());
                TextIndexOutput.htmlIndex = TextIndex.getHTML();
                formTextInd.setVisible(true);
            }

        }
    }

    class WindowCloser extends WindowAdapter {

        public void windowClosing(WindowEvent e) {
            Window win = e.getWindow();
            //win.setVisible(false);
            System.exit(0);
        }
    }
}
